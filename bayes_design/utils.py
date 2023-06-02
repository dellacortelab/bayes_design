import os
import torch
from Bio import pairwise2

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
    res_ids = pdb_dict_list[0]['res_ids']
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    protein = dataset_valid[0]
    struct = torch.tensor([protein['coords_chain_A']['N_chain_A'], protein['coords_chain_A']['CA_chain_A'], protein['coords_chain_A']['C_chain_A'], protein['coords_chain_A']['O_chain_A']]).transpose(0, 1)
    return protein['seq'], struct, res_ids

def get_fixed_position_mask(fixed_position_list, res_ids):

    full_fixed_position_list = [range(fixed_position_list[i], fixed_position_list[i+1]+1) for i in range(0, len(fixed_position_list), 2)]
    full_fixed_position_list = set([item for sublist in full_fixed_position_list for item in sublist])

    # Masked positions are the positions to predict/design
    # Default to no fixed positions, and thus predict all positions
    fixed_position_mask = np.zeros(len(res_ids))

    # Preserve fixed positions
    for i, id in enumerate(res_ids):
        if id in full_fixed_position_list:
            fixed_position_mask[i] = 1

    return fixed_position_mask


def structure_to_aligned_structure(seq, struct, aligned_seq):
    unaligned_idx = 0
    new_struct = torch.zeros(len(aligned_seq), 4, 3)
    for i in range(len(aligned_seq)):
        if aligned_seq[i] == '-':
            # could be a missing residue in the source sequence
            # could be a misalignment at the beginning or end
            new_struct[i] = torch.zeros(4, 3) * float('nan')
        else:
            new_struct[i] = struct[unaligned_idx]
            unaligned_idx += 1
    
    return new_struct

def align_and_crop(seq_pro, seq_anti, struct_pro, struct_anti):
    # Perform sequence alignment. Assumes the only differences are in some residues substituted with '-' in
    # one sequence but not the other
    alignments = pairwise2.align.globalxx(seq_pro, seq_anti, penalize_extend_when_opening=True)
    aligned_seq_pro_tmp, aligned_seq_anti_tmp, score, start, end = alignments[0]
    
    aligned_seq_pro = ''
    aligned_seq_anti = ''
    for i in range(len(aligned_seq_pro_tmp)):
        # If a dash exists at the same position in both sequences, remove it
        if aligned_seq_pro_tmp[i] == '-' and aligned_seq_anti_tmp[i] == '-':
            continue 
        aligned_seq_pro += aligned_seq_pro_tmp[i]
        aligned_seq_anti += aligned_seq_anti_tmp[i]

    new_struct_pro = structure_to_aligned_structure(seq_pro, struct_pro, aligned_seq_pro)
    new_struct_anti = structure_to_aligned_structure(seq_anti, struct_anti, aligned_seq_anti)

    # Strip off beginning and end residues that are absent in the pro sequence
    n_beg = 0
    while aligned_seq_pro[n_beg] == '-':
        n_beg += 1
    n_end = 0
    while aligned_seq_pro[-(n_end+1)] == '-':
        n_end += 1

    # make sure to account for case when n_end == 0
    if n_end == 0:
        aligned_seq_pro = aligned_seq_pro[n_beg:]
        aligned_seq_anti = aligned_seq_anti[n_beg:]
        new_struct_pro = new_struct_pro[n_beg:]
        new_struct_anti = new_struct_anti[n_beg:]
    else:
        aligned_seq_pro = aligned_seq_pro[n_beg:-n_end]
        aligned_seq_anti = aligned_seq_anti[n_beg:-n_end]
        new_struct_pro = new_struct_pro[n_beg:-n_end]
        new_struct_anti = new_struct_anti[n_beg:-n_end]

    # Merge the two sequences, defaulting to the pro sequence if both sequences have a residue at a position
    merged_seq = ''
    for i in range(len(aligned_seq_pro)):
        if aligned_seq_pro[i] != '-':
            merged_seq += aligned_seq_pro[i]
        elif aligned_seq_anti[i] != '-':
            merged_seq += aligned_seq_anti[i]
        else:
            merged_seq += 'X'

    return aligned_seq_pro, aligned_seq_anti, merged_seq, new_struct_pro, new_struct_anti


def get_ball_mask(fixed_position_list, struct, res_ids, radius=8):
    """Take a list of fixed positions and return a ball mask including all residues 
    within 8 angstroms of the unfixed positions
    Args:
        fixed_position_list (list): List of fixed positions
        struct ((L x 4 x 3) np.array): Structure of protein
    """
    struct_ca = struct[:, 1]
    fixed_position_mask = get_fixed_position_mask(fixed_position_list, res_ids)
    # Set positions within 8 angstroms of unfixed_positions to 0
    ball_mask = np.ones(len(res_ids))
    unfixed_position_mask = fixed_position_mask == 0
    for i in range(len(res_ids)):
        if unfixed_position_mask[i]:
            neighbors_mask = np.linalg.norm(struct_ca - struct_ca[i], axis=-1) < radius
            ball_mask[neighbors_mask] = 0
    return ball_mask