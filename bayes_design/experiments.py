
import torch
import numpy as np
import math
import os
from PIL import Image

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict, TrRosettaWrapper
from bayes_design.utils import get_protein, get_cb_coordinates, compute_distogram, AMINO_ACID_ORDER


def compare_seq_probs(args):
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
    prob_model = model_dict[args.model_name](device=device)

    _, structure = get_protein(args.protein_id)
    
    for seq in args.sequences:
        masked_positions = np.ones(len(seq))
        # Preserve fixed positions
        for i in range(0, len(args.fixed_positions), 2):
            # -1 because residues are 1-indexed
            fixed_range_start = args.fixed_positions[i] - 1
            # -1 because residues are 1-indexed and +1 because we are including the endpoint
            fixed_range_end = args.fixed_positions[i+1]
            masked_positions[fixed_range_start:fixed_range_end] = 0.
        masked_seq = ''.join(['-' if mask else char for char, mask in zip(seq, masked_positions)])
        n_masked_positions = int(masked_positions.sum())
        n_unmasked_positions = len(seq) - n_masked_positions

        # Decode order defines the order in which the masked positions are predicted
        decode_order = decode_order_dict[args.decode_order](masked_seq)
        token_to_decode = torch.tensor(decode_order[n_unmasked_positions:])
        input_seqs = [seq]*n_masked_positions

        probs = prob_model(seq=input_seqs, struct=structure, decode_order=decode_order, token_to_decode=token_to_decode)
        print(probs.shape)
        log_prob = 0
        for i, idx in enumerate(token_to_decode):
            aa = seq[idx]
            seq_idx = AMINO_ACID_ORDER.index(aa)
            log_prob += math.log(probs[i, seq_idx])
        
        print(f"Log Prob: {log_prob}, Sequence: {seq}")


def compare_struct_probs(args):

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
