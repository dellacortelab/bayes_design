
import torch
import numpy as np
import math
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.evaluate import metric_dict
from bayes_design.model import model_dict, TrRosettaWrapper
from bayes_design.utils import get_fixed_position_mask, get_protein, get_cb_coordinates, compute_distogram, AMINO_ACID_ORDER

def compare_seq_metric(args):
    seq, structure = get_protein(args.protein_id)
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(seq))
    
    prob_model = model_dict[args.model_name](device=device)
    for seq in args.sequences:
        metric = metric_dict[args.metric](seq=seq, prob_model=prob_model, decode_order=args.decode_order, structure=structure, fixed_position_mask=fixed_position_mask)
        print(f'Metric {args.metric}:', metric)

def compare_struct_probs(args):
    """Return log p(struct=x|seq=s) for the given sequence and structure using trRosetta.
    """
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    # Get true distances
    cb_coordinates_true = get_cb_coordinates(args.protein_id)
    cb_distogram_true = compute_distogram(cb_coordinates_true)
    cb_distogram_true_idx = torch.argmax(cb_distogram_true, dim=-1)

    from PIL import Image
    img = cb_distogram_true_idx.float().cpu().detach().numpy()
    img_1 = np.round(img/img.max()*255)
    im = Image.fromarray(img_1)
    im.convert('RGB').save(os.path.join(args.results_dir, f'{args.protein_id}_trrosetta.jpg'))
    forward_struct_model = TrRosettaWrapper(device=device)

    # Get trRosetta-predicted distogram for designed sequences
    for seq in args.sequences:
        seq_id = args.protein_id + '_' + seq[:10]
        cb_distogram_predicted = torch.tensor(forward_struct_model(seq, seq_id=seq_id))
        # Edit predicted distogram to correspond to a more sensible, ordered layout, with not-in-contact next to the largest distances
        cb_distogram_predicted = torch.cat((cb_distogram_predicted[..., 1:], cb_distogram_predicted[..., :1]), dim=-1)
        # Make the diagonal entries correspond to the lowest distance bin, not the not-in-contact bin
        cb_distogram_predicted[:, :, 0].fill_diagonal_(1)
        for i in range(1, 36):
            cb_distogram_predicted[:, :, i].fill_diagonal_(0)
        
        # Get probability of true structure under predicted distograms.
        probs = torch.gather(input=cb_distogram_predicted.cpu(), dim=-1, index=cb_distogram_true_idx.unsqueeze(-1))
        log_probs = torch.log(probs)
        log_prob = torch.sum(log_probs)

        print("Sequence", seq)
        print("Prob:", log_prob)

        img = torch.argmax(cb_distogram_predicted, dim=-1).float().cpu().detach().numpy()
        img_1 = np.round(img/img.max()*255)
        im = Image.fromarray(img_1)
        im.convert('RGB').save(os.path.join(args.results_dir, f'{seq_id}_trrosetta.jpg'))
        
def compare_probs(struct_to_seq_model, seq_model, struct, seq, decode_order, bayes_balance_factor=0.):
    """Compare probability of residues under several models and return the distributions p(seq|struct), p(seq), 
    and p(struct|seq) over each amino acid in the sequence
    """
    probs = []
    current_seq = seq
    for i in range(len(seq)):
        # Decode this item last, so that there is a full bidirectional mlm context for xlnet and protein_mpnn
        decode_order = np.append(np.delete(np.arange(len(seq)), i), i)
        p_seq_struct = struct_to_seq_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=i).clone()
        p_seq = seq_model([current_seq], decode_order=decode_order, token_to_decode=i, mask_type='bidirectional_mlm').clone()
        p_seq_struct_div_p_seq = p_seq_struct / p_seq
        p_struct_seq = p_seq_struct_div_p_seq / p_seq_struct_div_p_seq.sum()
        # For non-zero balance factor, return probabilities associated with balanced data
        if bayes_balance_factor != 0:
            p_seq_struct += bayes_balance_factor
            p_seq += bayes_balance_factor
            p_seq_struct_div_p_seq = p_seq_struct / p_seq
            p_struct_seq = p_seq_struct_div_p_seq / p_seq_struct_div_p_seq.sum()
        probs.append((p_seq_struct, p_seq, p_struct_seq))

    return probs

def viz_probs(args):
    """Compare p(seq|struct)/p(seq), p(seq), and p(seq|struct), across all residues, highlighting top 1 in blue, second in green, third in red
    """
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    protein_mpnn = model_dict['protein_mpnn'](device=device)
    xlnet = model_dict['xlnet'](device=device)

    seq, structure = get_protein(args.protein_id)
    
    fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(seq))
    masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(seq, fixed_position_mask)])
    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)

    if args.from_scratch:
        # Mask the sequence only if designing a sequence from scratch
        seq = masked_seq
    else:
        pass

    probs = compare_probs(struct_to_seq_model=protein_mpnn, seq_model=xlnet, struct=structure, seq=seq, decode_order=decode_order, bayes_balance_factor=args.bayes_balance_factor)
    
    n_figures = 5
    # Plot 5 evenly-spaced probabilities
    spacing = len(args.sequence) // (n_figures - 1)
    indices = np.arange(0, len(args.sequence), spacing)
    
    fig, ax = plt.subplots(n_figures, figsize=(23, 7))

    def color_top_scores(plot, probs):
        # Color top three scores
        first, second, third = np.argsort(probs)[::-1][:3]
        plot.patches[first].set_facecolor('blue')
        plot.patches[second].set_facecolor('green')
        plot.patches[third].set_facecolor('red')
        
        
    labels = list(AMINO_ACID_ORDER[:-1])
    x = np.arange(len(labels))  # the label locations
    width = 0.1
    for i, idx in enumerate(indices):
        ax[i].set_xticks(x, labels=labels)
        p_seq_struct, p_seq, p_struct_seq = probs[idx]
        p_seq_struct, p_seq, p_struct_seq = p_seq_struct[0].detach().cpu().numpy(), p_seq[0].detach().cpu().numpy(), p_struct_seq[0].detach().cpu().numpy()
        rects_1 = ax[i].bar(x - 1.5*width, p_seq_struct, width, color='orange')
        color_top_scores(plot=rects_1, probs=p_seq_struct)
        rects_2 = ax[i].bar(x, p_seq, width, color='orange')
        color_top_scores(plot=rects_2, probs=p_seq)
        rects_3 = ax[i].bar(x + 1.5*width, p_struct_seq, width, color='orange')
        color_top_scores(plot=rects_3, probs=p_struct_seq)
        ax[i].set_ylabel("True residue: " + seq[idx])
    
    patch_1 = matplotlib.patches.Patch(color='orange', label='$left: p(seq|struct)$')
    patch_2 = matplotlib.patches.Patch(color='orange', label='$middle: p(seq)$')
    patch_3 = matplotlib.patches.Patch(color='orange', label='$right: p(struct|seq)$')
    plt.gcf().legend(handles=[patch_1, patch_2, patch_3], loc='upper right')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(args.results_path)

#  Experiment: 
# Take a structure with a known sequence. Use p(seq|struct) and p(seq|struct)/p(seq) 
# to design new sequences for the structure. Compare the log-probability and stability
# of the original structure and the two designed structures. In the best case, we
# will see, from most stable to least stable: p(struct|seq) = p(seq|struct)/p(seq) > orig_seq > p(seq|struct)

# Experiment:
# Same as previous experiment, but hold part of the sequence fixed.

# Experiment:
# Compare perplexity and recapitulation performance to rmsd and log_prob performance
# for p(seq|struct)/p(seq) and p(seq|struct). We expect p(seq|struct) to outperform
# p(seq|struct)/p(seq) for perplexity and recapitulation, but expect p(seq|struct)/p(seq)
# to outperform p(seq|struct) in rmsd and log_prob. We argue that perplexity and
# recapitulation are the wrong metrics for sequence design.
