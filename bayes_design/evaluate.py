
"""
Evaluation for Seq Design Models that maximize p(struct=S|seq) - i.e. sequence
design models made to design a sequence that matches a backbone.
"""
import torch
import numpy as np
import math

from bayes_design.decode import decode_order_dict
from bayes_design.model import model_dict
from bayes_design.utils import get_protein, AMINO_ACID_ORDER


def evaluate_log_prob(seq, prob_model, decode_order, fixed_position_mask, mask_type, structure=None):
    """
    Evaluate the log probability of the structure for the sequence under a 
    sequence to structure model. This measures p(struct|seq) for a designed sequence.
    We expect p(seq|struct)/p(seq) to be better than p(seq|struct) because 
    p(seq|struct)/p(seq) seeks to model p(struct|seq). This may be a flawed metric
    because the seq-to-struct network may only key in on a few important residues 
    for global structure, possibly failing to correctly model the probabilities of
    residues that are less important for folding.

    You should call evaluate_log_prob with a sequence designed with p(seq|struct)
    and with a sequence designed with p(seq|struct)/p(seq) and compare the log
    probabilities.

    If fixed positions are provided, they are excluded from the log probability calculation.
    
    Args:
        sequence (str): a string (no spaces) representing an amino acid sequence
        seq_to_struct_model (nn.Module): a network taking as input a 
    """
    masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(seq, fixed_position_mask)])
    n_fixed_positions = int(fixed_position_mask.sum())
    n_predicted_positions = len(seq) - n_fixed_positions

    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[decode_order](masked_seq)
    token_to_decode = torch.tensor(decode_order[n_fixed_positions:])

    probs = prob_model(seq=[seq]*n_predicted_positions, struct=structure, decode_order=decode_order, token_to_decode=token_to_decode, mask_type=mask_type)
    
    log_prob = 0
    for i, tok in enumerate(token_to_decode):
        aa = seq[tok]
        seq_idx = AMINO_ACID_ORDER.index(aa)
        log_prob += math.log(probs[i, seq_idx])
    
    return log_prob


def evaluate_rmsd(sequence, seq_to_struct_model, targ_struct):
    """
    Evaluate RMSD between the "true structure" (computationally predicted from 
    the true sequence) and the predicted structure for a designed sequence. This 
    is a good metric for evaluating p(struct|seq), but has a performance ceiling.
    You can only match the RMSD. Ideally, we want to design a sequence with RMSD
    as close to 0 as possible, but with higher stability than the original sequence
    (measured by evaluate_log_prob).
    """

def evaluate_perplexity(seq, prob_model, decode_order, structure=None, fixed_position_mask=None, mask_type=None):
    """
    Evaluate the perplexity of the sequence under the model. If fixed positions are provided, they
    are excluded from the perplexity calculation.
    """
    log_prob = evaluate_log_prob(seq=seq, prob_model=prob_model, decode_order=decode_order, structure=structure, fixed_position_mask=fixed_position_mask, mask_type=mask_type)
    n_fixed_positions = fixed_position_mask.sum()
    n = len(seq) - n_fixed_positions
    perplexity = np.exp(-1/n*log_prob)
    return perplexity

def evaluate_recapitulation(pred_seq, orig_seq):
    """
    Evaluate % recapitulation of the original sequence. This is a bad metric for 
    measuring p(struct|seq) because it really measures p(seq|struct).
    """
    perc_recapitulate = (predict_seq == orig_seq).mean()
    return perc_recapitulate

def evaluate_stability():
    """
    This is the best metric, as it measures p(struct|seq), without being subject
    to the approximations of using a computational model for p(struct|seq). But it
    must be based on experimental results.
    """
    pass


metric_dict = {'log_prob':evaluate_log_prob, 'perplexity':evaluate_perplexity}