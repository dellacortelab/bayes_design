
import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import pickle as pkl

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.evaluate import metric_dict
from bayes_design.model import model_dict, TrRosettaWrapper
from bayes_design.utils import get_fixed_position_mask, get_protein, get_cb_coordinates, compute_distogram, AMINO_ACID_ORDER


def compare_seq_metric(args):
    seq, structure = get_protein(args.protein_id)
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(seq))
    
    if args.redesign:
        mask_type = 'bidirectional_mlm'
    else:
        mask_type = 'bidirectional_autoregressive'
    
    if args.model_name == 'bayes_design':
        prob_model = model_dict[args.model_name](device=device, bayes_balance_factor=args.bayes_balance_factor)
    elif args.model_name == 'protein_mpnn':
        prob_model = model_dict[args.model_name](device=device)
    elif args.model_name == 'xlnet':
        prob_model = model_dict[args.model_name](device=device)
    elif args.model_name == 'pssm':
        prob_model = model_dict[args.model_name](args.pssm_path)

    scores = []    
    for seq in args.sequences:
        masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(seq, fixed_position_mask)])
        # Decode order defines the order in which the masked positions are predicted
        decode_order = decode_order_dict[args.decode_order](masked_seq)
        metric = metric_dict[args.metric](seq=seq, prob_model=prob_model, decode_order=decode_order, structure=structure, fixed_position_mask=fixed_position_mask, mask_type=mask_type)
        scores.append(metric)
    
    return scores
    
        

def generate_sequence_variants(orig_seq, num_variants=10, perc_residues_to_mutate=0.1):
    """Generate variants of orig_seq by mutating a random subset of residues
    Args:
        orig_seq (str): Original sequence
        num_variants (int): Number of variants to generate
        perc_residues_to_mutate (float): Percentage of residues to mutate
    Returns:
        variants (list): List of variants
    """
    variants = []
    for _ in range(num_variants):
        num_residues_to_mutate = int(perc_residues_to_mutate * len(orig_seq))
        residues_to_mutate = np.random.choice(len(orig_seq), num_residues_to_mutate, replace=False)
        variant = list(orig_seq)
        for residue in residues_to_mutate:
            variant[residue] = np.random.choice([aa for aa in AMINO_ACID_ORDER[:-1] if aa != orig_seq[residue]])
        variants.append(''.join(variant))
    return variants

def compare_struct_correlation(args):
    """Compute the correlation of model scores with the trRosetta probability. If comparing BayesDesign to ProteinMPNN,
    this must be for multiple sequences across a single structure. This is because BayesDesign is expected to correlate
    with trRosetta across multiple sequences for the same structure, but not across structures.
    """
    np.random.seed(0)
    sequence, structure = get_protein(args.protein_id)
    device = torch.device(args.device)
    fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(sequence))
    masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(sequence, fixed_position_mask)])
    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)
    
    mask_type = 'bidirectional_mlm'

    protein_mpnn = model_dict['protein_mpnn'](device=device)
    bayes_design = model_dict['bayes_design'](device=device)
    trRosetta = model_dict['trRosetta'](device=device)
        
    log_p_bayes = []
    log_p_protein_mpnn = []
    log_p_trRosetta = []
    sequence_variants = generate_sequence_variants(orig_seq=sequence, num_variants=args.num_variants, perc_residues_to_mutate=args.perc_residues_to_mutate)
    for seq in sequence_variants:
        log_p_protein_mpnn.append(metric_dict['log_prob'](seq=seq, prob_model=protein_mpnn, decode_order=args.decode_order, structure=structure, fixed_position_mask=fixed_position_mask, mask_type=mask_type))
        log_p_bayes.append(metric_dict['log_prob'](seq=seq, prob_model=bayes_design, decode_order=args.decode_order, structure=structure, fixed_position_mask=fixed_position_mask, mask_type=mask_type))
        log_p_trRosetta.append(get_trRosetta_log_prob(trRosetta=trRosetta, sequence=seq, protein_id=args.protein_id))
        
    # Make a matplotlib scatter plot of BayesDesign and trRosetta, with the Pearson correlation coefficient as the title
    fig, ax = plt.subplots()
    ax.scatter(log_p_bayes, log_p_trRosetta)
    ax.set_xlabel('BayesDesign')
    ax.set_ylabel('trRosetta')
    ax.set_title(f'Pearson correlation coefficient: {np.corrcoef(log_p_bayes, log_p_trRosetta)[0, 1]:.3f}')
    fig.savefig(os.path.join(args.results_dir, f'{args.protein_id}_bayes_trRosetta_scatter.png'))
    plt.close(fig)

    # Make a matplotlib scatter plot of ProteinMPNN and trRosetta, with the Pearson correlation coefficient as the title
    fig, ax = plt.subplots()
    ax.scatter(log_p_protein_mpnn, log_p_trRosetta)
    ax.set_xlabel('ProteinMPNN')
    ax.set_ylabel('trRosetta')
    ax.set_title(f'Pearson correlation coefficient: {np.corrcoef(log_p_protein_mpnn, log_p_trRosetta)[0, 1]:.3f}')
    fig.savefig(os.path.join(args.results_dir, f'{args.protein_id}_protein_mpnn_trRosetta_scatter.png'))
    plt.close(fig)


def make_pssm(args):
    sequences_path = args.sequences_path
    pssm_path = args.pssm_path
    designed_seqs = []
    with open(sequences_path, 'r') as f:
        for line in f:
            designed_seqs.append(line.strip())

    L = len(designed_seqs[0])
    char_to_index_dict = {char:i for i, char in enumerate(AMINO_ACID_ORDER)}
    # Calculate position-specific scoring matrix (PSSM) for designed sequences
    pssm_cnt_matrix = np.zeros((L, 20))
    
    for i, designed_seq in enumerate(designed_seqs):
        for j, char in enumerate(designed_seq):
            pssm_cnt_matrix[j, char_to_index_dict[char]] += 1

    pssm = pssm_cnt_matrix / len(designed_seqs)
    with open(pssm_path, 'wb') as f:
        pkl.dump(pssm, f)

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


def get_trRosetta_log_prob(trRosetta, sequence, protein_id):
    """Get the log probability of a ground truth structure for a given sequence by computing the trRosetta probability
    of that structure given the sequence

    Args:
        trRosetta (nn.Module): a trRosetta model
        sequence (str): a protein sequence
        protein_id (str): the protein ID of the protein sequence
    Returns:
        log_prob (float): the log probability of the structure for the given sequence
    """
    # Get true distances
    cb_coordinates_true = get_cb_coordinates(protein_id)
    cb_distogram_true = compute_distogram(cb_coordinates_true)
    cb_distogram_true_idx = torch.argmax(cb_distogram_true, dim=-1)
    
    with torch.no_grad():
        # Get trRosetta-predicted distogram for sequence
        cb_distogram_predicted = torch.tensor(trRosetta(sequence, seq_id=protein_id))
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

    return log_prob

def compare_probs(struct_to_seq_model, seq_model, struct, seq, decode_order, bayes_balance_factor=0., mask_type='bidirectional_mlm'):
    """Compare probability of residues under several models and return the distributions p(seq|struct), p(seq), 
    and p(struct|seq) over each amino acid in the sequence
    """
    probs = []
    # TODO: remove for loop, pass in batch of sequences
    for i in range(len(seq)):
        p_seq_struct = struct_to_seq_model(seq=[seq], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([i]), mask_type=mask_type).clone()
        p_seq = seq_model([seq], decode_order=decode_order, token_to_decode=torch.tensor([i]), mask_type=mask_type).clone()
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

def make_hist(args):
    # Take a list of sequences
    # Evaluate the probability of each sequence under the model
    # Make a histogram of the probabilities
    scores = compare_seq_metric(args)
    plt.hist(scores, bins=100)
    plt.xlabel(f'{args.metric}')
    plt.savefig(os.path.join(args.results_dir, f'{args.model_name}_{args.metric}_{args.protein_id}_hist.png'))

def seq_filter(args):
    scores = compare_seq_metric(args)
    seqs = args.sequences
    top = sorted(zip(scores, seqs), reverse=True)[:args.n_seqs]
    top_sequences_path = os.path.splitext(args.sequences_path)[0] + '_top.txt'
    with open(top_sequences_path, 'w') as f:
        for score, seq in top:
            f.write(f'{seq}\n')
    print(top)



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
