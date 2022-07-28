import os
import torch
from protein_mpnn_utils import parse_PDB, StructureDatasetPDB

from Bio.PDB import PDBParser
import numpy as np

AMINO_ACID_ORDER = 'ACDEFGHIKLMNPQRSTVWYX'

def get_protein(pdb_code='6MRR', structures_dir='/data/structures'):
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


def get_cb_coordinates(pdb_code, structures_dir='/data/structures'):
    """Gets the coordinates of the primary four atoms for each residue. Returns an
    L x 4 x 3 array, with the atoms in the following order: N, CA, C, CB. For glycine,
    provides CA coordinates in place of CB coordinates.
    Args:
        pdbfile (str): the full file path of a protein structure file in PDB format
    Returns:
        ((L x 4 x 3) torch.Tensor): the 3D coordinates of the primary 4 atoms in each 
            amino acid in the sequence
    """
    pdb_path = os.path.join(structures_dir, pdb_code + '.pdb')
    residues = list(PDBParser(PERMISSIVE=True, QUIET=True).get_structure(id=os.path.basename(pdb_path), file=pdb_path)[0].get_residues())

    L = len(residues)
    cb_coordinates = np.zeros((L, 3), dtype=np.float32)

    # Set the coordinates for every residue
    for i, residue in enumerate(residues):
        try:
            if residue.resname == 'GLY':
                cb_coordinates[i, :] = residue["CA"].get_coord()
            else:
                cb_coordinates[i, :] = residue["CB"].get_coord()
        except KeyError as e:
            cb_coordinates[i, :] = residue['Cb'.lower()].get_coord()

    return torch.tensor(cb_coordinates)

def compute_distance_matrix(coordinates, epsilon=0.):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 3) torch.Tensor): an array of all four major atom coordinates
            per residue
        epsilon (float): a term to stabilize the gradients (because backpropping through sqrt
            gives you NaN at 0)
    Returns:
        ((L x L) torch.Tensor): the distance matrix for the residues
    """
    # In reality, pred_coordinates is an output of the network, but we initialize it here for a minimal working example
    L = len(coordinates)
    gram_matrix = torch.mm(coordinates, torch.transpose(coordinates, 0, 1))
    gram_diag = torch.diagonal(gram_matrix, dim1=0, dim2=1)
    # gram_diag: L
    diag_1 = torch.matmul(gram_diag.unsqueeze(-1), torch.ones(1, L).to(coordinates.device))
    # diag_1: L x L
    diag_2 = torch.transpose(diag_1, dim0=0, dim1=1)
    # diag_2: L x L
    squared_distance_matrix = diag_1 + diag_2 - (2 * gram_matrix )
    distance_matrix = torch.sqrt( squared_distance_matrix + epsilon)
    return distance_matrix

def compute_bins(matrix, bins, include_less_than=False, include_greater_than=False):
    """Bin values based on the bins array. Works for distances and trRosetta features.
    Args:
        matrix ((L x n) torch.Tensor): the matrix to bin
        bins ((n_bins) array-like): the bin endpoints
        include_less_than (bool): whether to include a bin for less than the min value
        include_greater_than (bool): whether to include a bin for greater than the max value
    Returns:
        binned_matrix ((L x n x n_bins) torch.Tensor): the matrix, but binned
    """
    L, n = matrix.shape
    # Number of bins is based on whether we have a bin for less than the lowest and greater than the highest
    n_bins = len(bins) - 1 + include_less_than + include_greater_than
    
    # Populate distogram
    binned_matrix = torch.zeros((L, n, n_bins))

    if include_less_than:
        binned_matrix[:, :, 0] = matrix < bins[0]

    for i, (bin_min, bin_max) in enumerate(zip(bins[:-1], bins[1:])):
        # Bins are shifted by one if we have a "less than" bin
        binned_matrix[:, :, include_less_than + i] = ( (matrix >= bin_min) * (matrix < bin_max) )
    
    if include_greater_than:
        binned_matrix[:, :, -1] = matrix >= bins[-1]

    return binned_matrix

def compute_distogram(coordinates):
    """Compute the distance matrix for a tensor of the coordinates of the four major atoms
    Args:
        four_coordinates ((L x 4 x 3) np.ndarray): an array of all four major atom coordinates
            per residue
    Returns:
        ((N x L x L) torch.Tensor): the binned distance matrix for the atoms
    """
    distance_matrix = compute_distance_matrix(coordinates)
    # Make sure all distance values are positive
    assert torch.all(distance_matrix >= 0)
    # The endpoints of the bins (n_bins + 1 endpoints)
    tr_rosetta_bins = np.arange(2.5, 20.5, .5)
    # Compute the distogram
    distogram = compute_bins(matrix=distance_matrix, bins=tr_rosetta_bins, include_less_than=True, include_greater_than=True)
    # No need to normalize probabilities to sum to 1, because there is just one one in each distogram
    
    return distogram