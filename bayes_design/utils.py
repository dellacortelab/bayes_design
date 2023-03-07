import os
import torch
from .protein_mpnn.protein_mpnn_utils import parse_PDB, StructureDatasetPDB

import numpy as np

AMINO_ACID_ORDER = 'ACDEFGHIKLMNPQRSTVWYX'

def get_protein(pdb_code='6MRR', structures_dir='./data'):
    """Get a sequence in string format and 4-atom protein structure in L x 4 x 3
    tensor format (with atoms in N CA CB C order).
    """
    pdb_path = os.path.join(structures_dir, pdb_code + '.pdb')
    if not os.path.exists(pdb_path):
        os.system(f"cd {structures_dir} && wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    chain_list = ['A']
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    protein = dataset_valid[0]
    struct = torch.tensor([protein['coords_chain_A']['N_chain_A'], protein['coords_chain_A']['CA_chain_A'], protein['coords_chain_A']['C_chain_A'], protein['coords_chain_A']['O_chain_A']]).transpose(0, 1)
    return protein['seq'], struct

def get_fixed_position_mask(fixed_position_list, seq_len):
    # Masked positions are the positions to predict/design
    # Default to no fixed positions, and thus predict all positions
    fixed_position_mask = np.zeros(seq_len)
    # Preserve fixed positions
    for i in range(0, len(fixed_position_list), 2):
        # -1 because residues are 1-indexed
        fixed_range_start = fixed_position_list[i] - 1
        # -1 because residues are 1-indexed and +1 because we are including the endpoint
        fixed_range_end = fixed_position_list[i+1]
        fixed_position_mask[fixed_range_start:fixed_range_end] = 1.
    return fixed_position_mask


def align_and_crop(seq_pro, seq_anti, struct_pro, struct_anti):
    # Perform sequence alignment
    alignments = pairwise2.align.globalxx(seq_pro, seq_anti, penalize_extend_when_opening=True)
    aligned_seq_pro_tmp, aligned_seq_anti_tmp, score, start, end = alignments[0]
    
    # If a dash exists at the same position in both sequences, remove it
    aligned_seq_pro = ''
    aligned_seq_anti = ''
    for i in range(len(aligned_seq_pro_tmp)):
        if aligned_seq_pro_tmp[i] != '-' or aligned_seq_anti_tmp[i] != '-':
            aligned_seq_pro += aligned_seq_pro_tmp[i]
            aligned_seq_anti += aligned_seq_anti_tmp[i]    


    # Insert '-' characters into sequences and zeros into structures
    new_struct_pro = torch.zeros(len(aligned_seq_pro), 4, 3)
    new_struct_anti = torch.zeros(len(aligned_seq_anti), 4, 3)
    pro_idx = 0
    mismatch_cnt = 0

    for i in range(len(aligned_seq_pro)):
        try:
            # This takes care of the case when you reach the end of the original sequence
            orig_residue = seq_pro[pro_idx]
        except:
            new_struct_pro[i] = torch.zeros(4, 3) * float('nan')
            aligned_seq_pro = aligned_seq_pro[:i] + '-' + aligned_seq_pro[i+1:]
            continue
        if aligned_seq_pro[i] == '-' and orig_residue != '-' and mismatch_cnt <= 0:
            # Only situation where you don't increment pro_idx
            new_struct_pro[i] = torch.zeros(4, 3) * float('nan')
        elif aligned_seq_pro[i] != '-' and orig_residue == '-':
            new_struct_pro[i] = torch.zeros(4, 3) * float('nan')
            aligned_seq_pro = aligned_seq_pro[:i] + '-' + aligned_seq_pro[i+1:]
            pro_idx += 1
            mismatch_cnt += 1
        elif aligned_seq_pro[i] == '-' and orig_residue != '-' and mismatch_cnt > 0:
            new_struct_pro[i] = torch.zeros(4, 3) * float('nan')
            pro_idx += 1
            mismatch_cnt -= 1
        else:
            new_struct_pro[i] = struct_pro[pro_idx]
            pro_idx += 1

    anti_idx = 0
    mismatch_cnt = 0
    for i in range(len(aligned_seq_anti)):
        try:
            # fails if you reach the end of the sequence
            orig_residue = seq_anti[anti_idx]
        except:
            new_struct_anti[i] = torch.zeros(4, 3) * float('nan')
            aligned_seq_pro = aligned_seq_anti[:i] + '-' + aligned_seq_anti[i+1:]
            continue
        if aligned_seq_anti[i] == '-' and orig_residue != '-' and mismatch_cnt <= 0:
            # Only situation where you don't increment anti_idx
            new_struct_anti[i] = torch.zeros(4, 3) * float('nan')
        elif aligned_seq_anti[i] != '-' and orig_residue == '-':
            new_struct_anti[i] = torch.zeros(4, 3) * float('nan')
            aligned_seq_pro = aligned_seq_anti[:i] + '-' + aligned_seq_anti[i+1:]
            anti_idx += 1
            mismatch_cnt += 1
        elif aligned_seq_anti[i] == '-' and orig_residue != '-' and mismatch_cnt > 0:
            new_struct_anti[i] = torch.zeros(4, 3) * float('nan')
            anti_idx += 1
            mismatch_cnt -= 1
        else:
            new_struct_anti[i] = struct_anti[anti_idx]
            anti_idx += 1

    return aligned_seq_pro, aligned_seq_anti, new_struct_pro, new_struct_anti