####################################################################################
# Decode Orders and Algorithms
####################################################################################

from math import log
import random
import numpy as np
import torch
import itertools

from .utils import AMINO_ACID_ORDER
from .evaluate import metric_dict
from .model import BayesDesign

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

def decode_combinations(prob_model, struct, seq, unmasked_seq, decode_order, fixed_position_mask, from_scratch, exclude_aa=['C'], **kwargs):
    """Consider all combinations of 6 positions from a set of 19 given positions (indicated by the unfixed positions in the fixed position mask). Use an n to c decode order for each combination of positions. 
    Greedy decode. Select the three decoded sequences with the highest probability. Repeat for 5, 4, 3, 2, and 1 position. 
    """
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    n_seq_per_position = 3
    max_n_positions = 2
    
    unfixed_positions = [i for i, mask in enumerate(fixed_position_mask) if mask == False]

    # Greedy decode
    decoded_seqs = []
    seq_log_probs = []
    for n_positions in range(2, max_n_positions + 1):
        # Make a list of all combinations of n_positions from unfixed_positions
        combinations = list(itertools.combinations(unfixed_positions, n_positions))
        top_sequences = []
        top_probs = []
        for i, combination in enumerate(combinations):
            print(i)
            masked_seq = [*unmasked_seq]
            for position in combination:
                masked_seq[position] = '-'
            # Update fixed positions to include all positions except the masked positions
            fixed_position_mask = np.array([1 if char != '-' else 0 for char in masked_seq])
            decode_order = get_n_to_c_decode_order(masked_seq)
            orig_log_prob = metric_dict['log_prob'](seq=unmasked_seq, prob_model=prob_model, decode_order=decode_order, structure=struct, fixed_position_mask=fixed_position_mask, mask_type=mask_type, exclude_aa=exclude_aa)
            decoded_seq, new_log_prob = greedy_decode(prob_model=prob_model, struct=struct, seq=''.join(masked_seq), decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=from_scratch, exclude_aa=exclude_aa, return_prob=True, **kwargs)
            log_prob_diff = new_log_prob - orig_log_prob
            # if log_prob in the top n_seq_per_position, add it to the list
            if len(top_probs) < n_seq_per_position:
                top_probs.append(log_prob_diff)
                top_sequences.append(decoded_seq)
            elif log_prob_diff > min(top_probs):
                min_index = np.argmin(top_probs)
                top_probs[min_index] = log_prob_diff
                top_sequences[min_index] = decoded_seq
            print(log_prob_diff)
            print(top_sequences)
            print(top_probs)
        
        seq_log_probs.extend(top_probs)
        decoded_seqs.extend(top_sequences)

    return decoded_seqs

def max_prob_decode(prob_model, struct, seq, unmasked_seq, decode_order, fixed_position_mask, from_scratch, n_decoded=20, exclude_aa=['C'], **kwargs):
    # Select unfixed positions (e.g. 63-96)
    # Evaluate the probability of every different amino acid at each unfixed position
    # Select residue with highest margin between p(struct|seq_i=true_aa) and p(struct|seq_i=best_alternate_aa).
    # Recalculate the probability of every different amino at each unfixed position, condiioned on the previous selection.
    # Select the next residue based on the highest margin, etc.
    # Repeat n_decoded times
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    current_seq = [*seq]
    current_seq_unmasked = [*unmasked_seq]
    
    n_fixed_positions = int(sum(fixed_position_mask))
    n_unfixed_positions = len(seq) - n_fixed_positions
    if n_decoded > n_unfixed_positions:
        n_decoded = n_unfixed_positions

    seqs = []
    for i in range(n_decoded):
        decode_order = get_n_to_c_decode_order(current_seq)
        # These are the eligible positions to decode
        token_to_decode = decode_order[n_fixed_positions + i:]
        # Get probabilities for every amino acid for every eligible position
        probs = prob_model(seq=[''.join(current_seq)]*len(token_to_decode), struct=struct, decode_order=decode_order, token_to_decode=token_to_decode, mask_type=mask_type)
        for aa in exclude_aa:
            probs[:, AMINO_ACID_ORDER.index(aa)] = 0
        probs = probs / probs.sum(dim=1, keepdim=True)
        # Select the position and amino acid for which the difference in probability between the true amino acid and the best alternate amino acid is the largest
        token_to_decode_true_aa_indices = [AMINO_ACID_ORDER.index(unmasked_seq[j]) for j in token_to_decode]
        true_probs = probs[range(len(token_to_decode)), token_to_decode_true_aa_indices] # N
        alternate_probs = probs.clone() # N x 20
        # Exclude true probs
        alternate_probs[range(len(token_to_decode)), token_to_decode_true_aa_indices] = -1
        
        alternate_probs, alternate_aa_ids = torch.max(alternate_probs, dim=1) # N
        margin = alternate_probs - true_probs # N
        # Select the position and amino acid for which the difference in probability between the true amino acid and the best alternate amino acid is the largest
        position = torch.argmax(margin)
        next_position = token_to_decode[position]
        next_aa = AMINO_ACID_ORDER[alternate_aa_ids[position]]
        current_seq[next_position] = next_aa
        current_seq_unmasked[next_position] = next_aa
        seqs.append(''.join(current_seq_unmasked))

    return seqs

        
def greedy_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, exclude_aa=['C'], return_prob=False, **kwargs):
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
        for aa in exclude_aa:
            probs[0, AMINO_ACID_ORDER.index(aa)] = 0
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        next_item = torch.argmax(probs)
        log_probs.append(np.log(np.max(probs.detach().cpu().numpy())))
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)

    if return_prob:
        return current_seq, np.sum(log_probs)
    else:
        return current_seq

def sample_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, exclude_aa, temperature=1.0, **kwargs):
    assert not isinstance(prob_model, BayesDesign), "BayesDesign objective only applies when maximizing probability"
    if from_scratch:
        mask_type = 'bidirectional_autoregressive'
    else:
        mask_type = 'bidirectional_mlm'
    current_seq = seq
    for idx in decode_order:
        print("j:", idx)
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        probs = prob_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=torch.tensor([idx]), mask_type=mask_type, temperature=temperature).detach().cpu().numpy()
        for aa in exclude_aa:
            probs[0, AMINO_ACID_ORDER.index(aa)] = 0
        probs = probs / torch.sum(probs, dim=1, keepdim=True).detach().cpu().numpy()
        next_item = np.random.choice(np.arange(20), p=probs[0])

        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def random_decode(prob_model, struct, seq, decode_order, fixed_position_mask, exclude_aa, **kwargs):
    current_seq = seq
    eligible_aa_idxs = [i for i in range(20) if AMINO_ACID_ORDER[i] not in exclude_aa]
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        next_item = np.random.choice(eligible_aa_idxs)
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def beam_decode(prob_model, struct, seq, decode_order, fixed_position_mask, exclude_aa, from_scratch, n_beams, **kwargs):

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
        for aa in exclude_aa:
            top_candidate_probs[:, AMINO_ACID_ORDER.index(aa)] = 0
        top_candidate_probs = top_candidate_probs / torch.sum(top_candidate_probs, dim=1, keepdim=True)
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
decode_algorithm_dict = {'greedy':greedy_decode, 'beam':beam_decode, 'sample':sample_decode, 'random':random_decode, 'max_prob_decode':max_prob_decode, 'combinations':decode_combinations}
