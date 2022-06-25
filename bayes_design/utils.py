import os
import torch
from protein_mpnn_utils import parse_PDB, StructureDatasetPDB


AMINO_ACID_ORDER = 'ACDEFGHIKLMNPQRSTVWYX'

def get_protein(pdb_code='6MRR'):
    """Get a sequence in string format and 4-atom protein structure in L x 4 x 3
    tensor format (with atoms in N CA CB C order).
    """
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    pdb_path = f"{pdb_code}.pdb"
    chain_list = ['A']
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, max_length=20000)
    protein = dataset_valid[0]
    struct = torch.tensor([protein['coords_chain_A']['N_chain_A'], protein['coords_chain_A']['CA_chain_A'], protein['coords_chain_A']['C_chain_A'], protein['coords_chain_A']['O_chain_A']]).transpose(0, 1)
    return protein['seq'], struct
