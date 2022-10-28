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

def greedy_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    """
    This function works.
    """
    current_seq = seq
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        if not from_scratch:
            # Mask the current token being decoded
            current_seq = list(current_seq)
            current_seq[idx] = '-'
            current_seq = ''.join(current_seq)
        probs = prob_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=idx)
        next_item = torch.argmax(probs)
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def sample_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    """
    This function works
    """
    current_seq = seq
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        if not from_scratch:
            # Mask the current token being decoded
            current_seq = list(current_seq)
            current_seq[idx] = '-'
            current_seq = ''.join(current_seq)
        probs = prob_model(seq=[current_seq], struct=struct, decode_order=decode_order, token_to_decode=idx).detach().cpu().numpy()
        next_item = np.random.choice(np.arange(20), p=probs)
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def random_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch):
    """
    This function works.
    """
    current_seq = seq
    for idx in decode_order:
        if fixed_position_mask[idx] == True:
            # Do not change this token
            continue
        next_item = np.random.choice(np.arange(21))
        aa = AMINO_ACID_ORDER[next_item]
        current_seq = list(current_seq)
        current_seq[idx] = aa
        current_seq = ''.join(current_seq)
    return current_seq

def beam_decode(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, n_beams):
    """
    This function works.
    """    
    top_candidates = [[list(seq), 0.0]]
    for j, decode_idx in enumerate(decode_order):
        print("j:", j)
        # If token is fixed, select the fixed token, regardless of probability
        if fixed_position_mask[decode_idx] == True:
            continue
        all_candidates = []
        for (current_seq, score) in top_candidates:
            if not from_scratch:
                # Mask the current token being decoded
                current_seq = list(current_seq)
                current_seq[decode_idx] = '-'
                current_seq = ''.join(current_seq)
            probs = prob_model(seq=[''.join(current_seq)], struct=struct, decode_order=decode_order, token_to_decode=decode_idx)[0].tolist()
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
    return top_candidates[0]

def beam_decode_medium(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, n_beams):
    """
    This function works. TODO: Incorporate 
        if not from_scratch:
            # Mask the current token being decoded
            current_seq = list(current_seq)
            current_seq[idx] = '-'
            current_seq = ''.join(current_seq)
    """
    top_candidates = [[list(seq), 0.0]]
    for j, decode_idx in enumerate(decode_order):
        print("j:", j)
        # If token is fixed, select the fixed token, regardless of probability
        if fixed_position_mask[decode_idx] == True:
            continue
        top_sequences = [seq for seq, score in top_candidates]
        # if not from_scratch:
        #     for i in range(len(top_sequences)):
        #         top_sequences[i][decode_idx] = '-'
        #     # top_sequences[:][decode_idx] = '-'
        #     import pdb; pdb.set_trace()
        top_candidate_probs = prob_model(seq=[''.join(seq) for seq in top_sequences], struct=struct, decode_order=decode_order, token_to_decode=decode_idx)
        all_candidates = []
        for ((current_seq, score), next_aa_probs) in zip(top_candidates, top_candidate_probs):
            for i, prob in enumerate(next_aa_probs.tolist()):
                candidate_seq = current_seq.copy()
                candidate_seq[decode_idx] = AMINO_ACID_ORDER[i]
                candidate = [candidate_seq, score + log(prob)]
                all_candidates.append(candidate)
        # Order all candidates by log-prob (highest to lowest)
        ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
        # Select n_beams best
        top_candidates = ordered[:n_beams]
    top_candidates = [(''.join(seq), score) for (seq, score) in top_candidates]
    return top_candidates[0]

def beam_decode_fast(prob_model, struct, seq, decode_order, fixed_position_mask, from_scratch, n_beams):
    """
    This function does not yet work.
    """
    L = len(seq)
    top_candidates = [[list(seq), 0.0]]
    for j, decode_idx in enumerate(decode_order):
        print("j:", j)
        top_sequences = [seq for seq, score in top_candidates]
        # top_sequences: n_beams x L
        probs = prob_model(seq=[''.join(seq) for seq in top_sequences], struct=struct, decode_order=decode_order, token_to_decode=decode_idx)
        # probs: n_beams x 21
        proposed_sequences = np.array(top_sequences)
        proposed_sequences = np.repeat(proposed_sequences[:, np.newaxis, :], len(AMINO_ACID_ORDER), axis=1)
        proposed_sequences[:, np.arange(len(AMINO_ACID_ORDER)), decode_idx] = np.array(list(AMINO_ACID_ORDER))
        # proposed_sequences: n_beams x 21 x L

        if fixed_position_mask[decode_idx] == True:
            fixed_aa_idx = AMINO_ACID_ORDER.index(seq[decode_idx])
            probs = probs[:, fixed_aa_idx].unsqueeze(-1)
            # probs: n_beams x 1
            proposed_sequences = proposed_sequences[:, fixed_aa_idx, :][:, np.newaxis, :]
            # proposed_sequences: n_beams x 1 x L

        top_scores = torch.tensor([score for seq, score in top_candidates]).to(probs.device)
        # top_scores: n_beams
        summed_probs = top_scores[:, None] + torch.log(probs)
        # summed_probs: n_beams x 21
        orig_shape = summed_probs.shape
        summed_probs = summed_probs.reshape(-1)
        proposed_sequences = proposed_sequences.reshape(-1, proposed_sequences.shape[-1])
        top_k = torch.topk(summed_probs, k=min(n_beams, len(summed_probs)))
        top_probs = top_k.values.tolist()
        top_sequences = [proposed_sequences[i].tolist() for i in top_k.indices]
        top_candidates = [list(x) for x in zip(top_sequences, top_probs)]

    top_candidates = [(''.join(seq), score) for (seq, score) in top_candidates]
    return top_candidates[0]



decode_order_dict = {'proximity':get_proximity_decode_order, 'reverse_proximity':get_reverse_proximity_decode_order, 'random':get_random_decode_order, 'n_to_c':get_n_to_c_decode_order}
decode_algorithm_dict = {'greedy':greedy_decode, 'beam_fast':beam_decode_fast, 'beam_medium':beam_decode_medium, 'beam':beam_decode, 'sample':sample_decode, 'random':random_decode}
