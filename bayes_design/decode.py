####################################################################################
# Decode Orders and Algorithms
####################################################################################

from math import log
import random
import numpy as np
import torch

from .utils import AMINO_ACID_ORDER

####################################################################################
# Decode Orders
####################################################################################

def get_proximity_decode_order(seq):
    """Accept an amino acid sequence with fixed residues provided and other
    residues represented by a dash (-) character. Return a decoding order that
    prioritizes predicted residues based on sequence proximity to fixed residues.
    Args:
        seq (len L str): a string representation of an amino
            acid sequence with unknown residues indicated with a dash (-)
    Returns:
        decode_order (list of len L): a list where the integer in position 0
            indicates the first index to decode, the integer in position 1 indicates
            the next index to decode, etc.
    """
    all_indices = np.arange(len(seq))
    indices_to_predict = [i for i, char in zip(all_indices, seq) if char == '-']
    fixed_indices = [i for i, char in zip(all_indices, seq) if char != '-']
    distances = []
    for i in indices_to_predict:
        distance = min([abs(idx - i) for idx in fixed_indices])
        distances.append(distance)
    order = np.argsort(distances)
    indices_to_predict_sorted = [indices_to_predict[i] for i in order]
    decode_order = fixed_indices + indices_to_predict_sorted
    return decode_order

def get_reverse_proximity_decode_order(seq):
    """Accept an amino acid sequence with fixed residues provided and other
    residues represented by a dash (-) character. Return a decoding order that
    prioritizes predicted residues based on sequence distance to fixed residues.
    Args:
        seq (len L str): a string representation of an amino
            acid sequence with unknown residues indicated with a dash (-)
    Returns:
        decode_order (list of len L): a list where the integer in position 0
            indicates the first index to decode, the integer in position 1 indicates
            the next index to decode, etc.
    """
    all_indices = np.arange(len(seq))
    indices_to_predict = [i for i, char in zip(all_indices, seq) if char == '-']
    fixed_indices = [i for i, char in zip(all_indices, seq) if char != '-']
    distances = []
    for i in indices_to_predict:
        distance = min([abs(idx - i) for idx in fixed_indices])
        distances.append(distance)
    order = np.argsort(distances)
    indices_to_predict_sorted = [indices_to_predict[i] for i in order]
    decode_order = fixed_indices + reversed(indices_to_predict_sorted)
    return decode_order

def get_random_decode_order(seq):
    """Accept an amino acid sequence with fixed residues provided and other
    residues represented by a dash (-) character. Return a random decoding order.
    Args:
        seq (len L str): a string representation of an amino
            acid sequence with unknown residues indicated with a dash (-)
    Returns:
        decode_order (list of len L): a list where the integer in position 0
            indicates the first index to decode, the integer in position 1 indicates
            the next index to decode, etc.
    """
    all_indices = np.arange(len(seq))
    indices_to_predict = [i for i, char in zip(all_indices, seq) if char == '-']
    fixed_indices = [i for i, char in zip(all_indices, seq) if char != '-']
    decode_order = fixed_indices + random.shuffle(indices_to_predict)
    return decode_order

def get_n_to_c_decode_order(seq):
    """Accept an amino acid sequence with fixed residues provided and other
    residues represented by a dash (-) character. Return a decoding order from the
    n terminus to the c terminus.
    Args:
        seq (len L str): a string representation of an amino
            acid sequence with unknown residues indicated with a dash (-)
    Returns:
        decode_order (list of len L): a list where the integer in position 0
            indicates the first index to decode, the integer in position 1 indicates
            the next index to decode, etc.
    """
    all_indices = np.arange(len(seq))
    indices_to_predict = [i for i, char in zip(all_indices, seq) if char == '-']
    fixed_indices = [i for i, char in zip(all_indices, seq) if char != '-']
    decode_order = fixed_indices + indices_to_predict
    return decode_order


####################################################################################
# Decode Algorithms
####################################################################################

def greedy_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    current_seq = seq
    log_probs = []
    for i, idx in enumerate(decode_order):
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        probs = prob_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([idx]), mask_type=mask_type)
        next_item = torch.argmax(probs)
        log_probs.append(np.log(np.max(probs.detach().cpu().numpy())))
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    print("log probs:", log_probs)
    print("log prob:", np.array(log_probs).sum())
    return current_seq

def sample_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    current_seq = seq
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        probs = prob_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([idx]), mask_type=mask_type).detach().cpu().numpy()
        next_item = np.random.choice(np.arange(20), p=probs)
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def random_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    current_seq = seq
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        next_item = np.random.choice(np.arange(20))
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def beam_decode_slow(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, n_beams):
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    top_candidates = [[list(seq), 0.0]]
    for j, decode_idx in enumerate(decode_order):
        print("j:", j)
        # If token is fixed, select the fixed token, regardless of probability
        if fixed_position_mask[decode_idx] == True:
            continue
        all_candidates = []
        for (current_seq, score) in top_candidates:
            probs = prob_model(seq=[''.join(current_seq)], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([decode_idx]), mask_type=mask_type)[0].tolist()
            for i, prob in enumerate(probs):
                candidate_seq = current_seq.copy()
                candidate_seq[decode_idx] = AMINO_ACID_ORDER[i]
                candidate = [candidate_seq, score + log(prob)]
                all_candidates.append(candidate)
        # Order all candidates by log-prob (highest to lowest)
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        # Select n_beams best
        top_candidates = ordered[:n_beams]
    top_candidates = [(''.join(seq), score) for (seq, score) in top_candidates]
    return top_candidates[0][0]

def beam_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, n_beams):

    # Get device of prob_model
    device = next(prob_model.parameters()).device
    # Get available gpu memory
    gpu_mem = torch.cuda.get_device_properties(device).total_memory
    # Get the number of concurrent sequences that can fit on one GPU
    n_concurrent_seqs = int(196 / 32510 * (.9 * gpu_mem / 1e6))

    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    top_candidates = [[list(seq), [0.0]]]
    for j, decode_idx in enumerate(decode_order):
        print("j:", j)
        # If token is fixed, select the fixed token, regardless of probability
        if fixed_position_mask[decode_idx] == True:
            continue
        top_sequences = [seq for seq, score in top_candidates]
        seqs = [''.join(seq) for seq in top_sequences]
        # Chunk up predictions so they fit on one GPU
        probs_list = []
        for i in range(0, len(seqs), n_concurrent_seqs):
            probs = prob_model.forward(seq=seqs[i:i + n_concurrent_seqs], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([decode_idx]).expand(len(seqs[i:i + n_concurrent_seqs])), mask_type=mask_type)
            probs_list.append(probs)
        top_candidate_probs = torch.concat(probs_list, dim=0)
        all_candidates = []
        for ((current_seq, score), next_aa_probs) in zip(top_candidates, top_candidate_probs):
            for i, prob in enumerate(next_aa_probs.tolist()):
                candidate_seq = current_seq.copy()
                candidate_seq[decode_idx] = AMINO_ACID_ORDER[i]
                candidate = [candidate_seq, score + [log(prob)]]
                all_candidates.append(candidate)
        # Order all candidates by log-prob (highest to lowest)
        ordered = sorted(all_candidates, key=lambda tup:np.array(tup[1]).sum(), reverse=True)
        # Select n_beams best
        top_candidates = ordered[:n_beams]
    top_candidates = [(''.join(seq), score) for (seq, score) in top_candidates]
    return top_candidates[0][0]


decode_order_dict = {'proximity':get_proximity_decode_order, 'reverse_proximity':get_reverse_proximity_decode_order, 'random':get_random_decode_order, 'n_to_c':get_n_to_c_decode_order}
decode_algorithm_dict = {'greedy':greedy_decode, 'beam':beam_decode, 'beam_slow':beam_decode_slow, 'sample':sample_decode, 'random':random_decode}
