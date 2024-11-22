# scRMSD vs. ProteinMPNN in silico experiment

from Bio.PDB import PDBList, PDBParser
import torch
import numpy as np
import os
import argparse
from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.utils import get_protein, align_and_crop, get_ball_mask, get_fixed_position_mask
from bayes_design.experiments.cath import parse_cath_file

from tqdm import tqdm

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import seq1 as three_letter_to_one_letter
from Bio.PDB.Polypeptide import PPBuilder
import json

from collections import defaultdict
from torch.nn import functional as F


from bayes_design.utils import AMINO_ACID_ORDER
tokenizer_dict = {
    aa: i for i, aa in enumerate(AMINO_ACID_ORDER)
}
tokenizer_dict["-"] = tokenizer_dict["X"]
detokenizer_dict = {
    i: aa for i, aa in tokenizer_dict.items()
}

def custom_collate_fn(batch):
    """Pad sequences, coords, and res_ids to the maximum length in the batch.
    batch is a list of pdbs containing a list of chains containing a tuple of (coords, seq, res_ids)
    """
    max_coords_len = max(max(chain[0].shape[0] for chain in entry) for entry in batch)
    max_seq_len = max(max(len(chain[1]) for chain in entry) for entry in batch)
    max_res_id_len = max(max(len(chain[2]) for chain in entry) for entry in batch)
    assert max_seq_len == max_res_id_len == max_coords_len, "Sequence, res_id, and coords lengths must be the same."
    coord_batch = []
    seq_batch = []
    res_id_batch = []
    for entry in batch:
        for chain in entry:
            coords, seq, res_ids = chain
            # seq = F.pad(seq, ("X", max_seq_len - len(seq)), value=-1)
            seq = seq + ["X"]*(max_seq_len - len(seq))
            res_ids = F.pad(res_ids, (0, max_res_id_len - len(res_ids)), value=-1)
            coords = F.pad(coords, (0, 0, 0, max_coords_len - coords.shape[0]), value=-1)
            coord_batch.append(coords)
            seq_batch.append(seq)
            res_id_batch.append(res_ids)

    return torch.stack(coord_batch), seq_batch, torch.stack(res_id_batch)

# TODO: ignore "X" matches

def tokenize_sequence(sequence):
    return torch.tensor([tokenizer_dict[aa] for aa in sequence])

def detokenize_sequence(sequence):
    return "".join([detokenizer_dict[i] for i in sequence])

class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_dir, verbose=False):
        self.pdb_dir = pdb_dir
        self.pdb_paths = [os.path.join(root, file) for root, dirs, files in os.walk(pdb_dir) for file in files]
        self.parser = PDBParser(QUIET=not verbose)

    def __len__(self):
        return len(self.pdb_paths)
    
    def extract_seq_and_res_ids(self, chain):
        sequence = []
        res_ids = []
        coords = []
        for residue in chain:
            if residue.has_id("CA"):  # Filter to only include amino acids
                sequence.append(three_letter_to_one_letter(residue.get_resname()))
                res_ids.append(residue.id[1])
                coords.append(residue["CA"].get_coord())
        return coords, sequence, res_ids
    
    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0][3:]
        structure = self.parser.get_structure(pdb_id, pdb_path)

        chains = []
        for chain in structure[0]:
            coords, sequence, res_ids = self.extract_seq_and_res_ids(chain)
            coords = torch.tensor(coords)
            # sequence = tokenize_sequence(sequence)
            res_ids = torch.tensor(res_ids)

            chains.append((coords, sequence, res_ids))

        return chains


class CathDataset(PDBDataset):
    def __init__(self, pdb_dir, cath_domains):
        super().__init__(pdb_dir)
        self.cath_domains = cath_domains
        self.superfamily_groups = {}
        for domain in cath_domains:
            key = (domain.class_number, domain.architecture, domain.topology, domain.homologous_superfamily)
            if key not in self.superfamily_groups:
                self.superfamily_groups[key] = []
            self.superfamily_groups[key].append(domain)

    def get_domain_path(self, domain_name):
        pdb_id = domain_name[:4].lower()
        middle_chars = pdb_id[1:3]
        # TODO: Make sure this points to the right place and make sure that I can replicate the creation of this dataset
        return os.path.join(self.pdb_dir, middle_chars, f"pdb{pdb_id}.ent")

    def get_superfamily_domains(self, domain):
        key = (domain.class_number, domain.architecture, domain.topology, domain.homologous_superfamily)
        return self.superfamily_groups.get(key, [])
    
def batch_align_sequence_with_res_ids(seq1, res_ids1, seq2, res_ids2):
    """Batch variant of align_sequences_with_res_ids. Uses multiprocessing to align in parallel.
    Args:
        seq1 ((B) list of (N) list of str): A list of sequences, where each sequence is a list of amino acids.
        res_ids1 ((B, N) tensor): A tensor of residue ids corresponding to each amino acid in seq1.
        seq2 ((B) list of (M) list of str): A list of sequences, where each sequence is a list of amino acids.
        res_ids2 ((B, M) tensor): A tensor of residue ids corresponding to each amino acid in seq2.
    """
    import multiprocessing
    
    # create a pool of workers
    pool = multiprocessing.Pool()

    # align each sequence pair in parallel
    results = pool.starmap(align_sequences_with_res_ids, zip(seq1, res_ids1, seq2, res_ids2))

    # close the pool
    pool.close()

    return results

    



# Example function to align and map residue ids
def align_sequences_with_res_ids(seq1, res_ids1, seq2, res_ids2):
    """
    Args:
        seq1 ((N) list of str): A list of sequences, where each sequence is a list of amino acids.
        res_ids1 (N Tensor): A tensor of residue ids corresponding to each amino acid in seq1.
        seq2 ((M) list of str): A list of sequences, where each sequence is a list of amino acids.
        res_ids2 (M tensor): A tensor of residue ids corresponding to each amino acid in seq2.
    Returns:
        aligned_res_ids1 ((L) list of int): A list of residue ids corresponding to the aligned sequence of seq1.
        aligned_res_ids2 ((L) list of int): A list of residue ids corresponding to the aligned sequence of seq2.
        aligned_seq1 ((L) list of str): The aligned sequence of seq1.
        aligned_seq2 ((L) list of str): The aligned sequence
    """
    res_ids1 = res_ids1.tolist()
    res_ids2 = res_ids2.tolist()
    # Align sequences
    seq1 = "".join(seq1)
    seq2 = "".join(seq2)
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = alignments[0]  # Use the best alignment

    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
    
    # Create mappings of aligned positions to res_ids
    aligned_res_ids1 = []
    aligned_res_ids2 = []
    
    idx1, idx2 = 0, 0
    for i in range(len(aligned_seq1)):
        if aligned_seq1[i] != '-':
            aligned_res_ids1.append(res_ids1[idx1])
            idx1 += 1
        else:
            aligned_res_ids1.append(None)

        if aligned_seq2[i] != '-':
            aligned_res_ids2.append(res_ids2[idx2])
            idx2 += 1
        else:
            aligned_res_ids2.append(None)

    # truncate beginning and end if missing residues on either sequence
    beginning_idx = 0
    while aligned_res_ids1[beginning_idx] is None or aligned_res_ids2[beginning_idx] is None:
        beginning_idx += 1
    end_idx = len(aligned_res_ids1) - 1
    while aligned_res_ids1[end_idx] is None or aligned_res_ids2[end_idx] is None:
        end_idx -= 1

    aligned_res_ids1 = aligned_res_ids1[beginning_idx:end_idx+1]
    aligned_res_ids2 = aligned_res_ids2[beginning_idx:end_idx+1]
    aligned_seq1 = aligned_seq1[beginning_idx:end_idx+1]
    aligned_seq2 = aligned_seq2[beginning_idx:end_idx+1]

    return aligned_res_ids1, aligned_res_ids2, aligned_seq1, aligned_seq2


def get_sequence(chain):
    """Extract the amino acid sequence of a chain."""
    ppb = PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(chain):
        seq += seq1(pp.get_sequence())
    return seq

def align_sequences(seq1, seq2):
    """Perform global alignment between two sequences."""
    alignments = pairwise2.align.globalxx(seq1, seq2)
    best_alignment = max(alignments, key=lambda x: x[2])  # Get alignment with highest score
    # Jointly 
    return best_alignment

def calculate_identity(align1, align2):
    """Calculate sequence identity between two aligned sequences."""
    assert len(align1) == len(align2)
    matches = sum(res1 == res2 for res1, res2 in zip(align1, align2))
    return matches / len(align1) * 100

def compute_windowed_sequence_identity(seq1, seq2, window_size=10):
    """Compute the sequence identity between two sequences in a sliding window fashion.
    Args:
        seq1 (list of int of length N): A list of residue indices for the first sequence.
        seq2 (list of int of length N): A list of residue indices for the second sequence.
        window_size (int): The size of the sliding window.
    Returns:
        sequence_identity (list of float of length N - window_size + 1): The sequence identity for each window.
    """
    seq1_tokenized = tokenize_sequence(seq1)
    seq2_tokenized = tokenize_sequence(seq2)

    # Ensure sequences are of the same length and convert them to tensors
    assert len(seq1_tokenized) == len(seq2_tokenized), "Sequences must be of the same length"
    
    # Compute the binary match tensor (1 if residues match, 0 if they don't)
    match_tensor = (seq1_tokenized == seq2_tokenized).float()
    # Set the match tensor to 0 if either residue is a gap
    match_tensor[(seq1_tokenized == tokenizer_dict["-"]) | (seq2_tokenized == tokenizer_dict["-"])] = 0
    
    # Compute the cumulative sum of matches
    cumulative_sum = torch.cumsum(match_tensor, dim=0)
    
    # Calculate the match count for each 10-residue window
    window_sums = cumulative_sum[window_size - 1:] - torch.cat((torch.tensor([0]), cumulative_sum[:-window_size]))
    
    # Calculate sequence identity for each window
    sequence_identity = window_sums / window_size
    
    return sequence_identity



def kabsch_alignment(coords1, coords2):
    # Step 1: Input validation
    assert coords1.shape == coords2.shape, "Input coordinate arrays must have the same shape."
    assert coords1.shape[1] == 3, "Coordinate arrays must have shape N x 3."
    
    # Step 2: Compute centroids of coords1 and coords2
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    # Step 3: Subtract the centroids
    coords1_centered = coords1 - centroid1
    coords2_centered = coords2 - centroid2
    
    # Step 4: Calculate the covariance matrix
    covariance_matrix = np.dot(coords2_centered.T, coords1_centered)
    
    # Step 5: Perform SVD on the covariance matrix
    U, S, Vt = np.linalg.svd(covariance_matrix)
    
    # Step 6: Compute the rotation matrix R
    d = np.linalg.det(np.dot(U, Vt))
    rotation_matrix = np.dot(U, np.dot(np.diag([1, 1, d]), Vt))
    
    # Step 7: Rotate the second set of coordinates (coords2)
    coords2_aligned = np.dot(coords2_centered, rotation_matrix)
    
    # Step 8: Translate rotated coords2 by adding back centroid of coords1
    coords2_aligned += centroid1
    
    # Step 9: Return the aligned coordinates
    return coords2_aligned

def test_kabsch_alignment():
    coords1 = np.random.rand(10, 3)
    coords2 = np.copy(coords1)

    # Construct a general 3d rotation matrix
    alpha = np.pi / 4
    beta = np.pi / 3
    gamma = np.pi / 6
    rot = np.array([
        [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma), np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)],
        [np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma), np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)],
        [-np.sin(beta), np.cos(beta) * np.sin(gamma), np.cos(beta) * np.cos(gamma)]
    ])
    trans = np.random.rand(3)

    coords2 = np.dot(rot, coords2.T).T + trans

    coords2_aligned = kabsch_alignment(coords1, coords2)
    
    assert not np.allclose(coords1, coords2)
    assert np.allclose(coords1, coords2_aligned)

def test_compute_windowed_sequence_identity():
    seq1 = [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5]
    seq2 = [1, 2, 0, 1, 2, 3, 4, 1, 0, 3, 1, 0, 3, 4, 5]
    window_size = 10
    identity = compute_windowed_sequence_identity(seq1, seq2, window_size)
    assert np.allclose(identity, [0.8, 0.8, 0.7, 0.8, 0.8, 0.8])

def calculate_rmsd(chain_residues, new_chain_residues):
    """Calculate the RMSD between two chains.
    Args:
        chain_residues (list of Bio.PDB.Residue): The residues of the first chain.
        new_chain_residues (list of Bio.PDB.Residue): The residues of the second chain.
    Returns:
        rmsd (float): The RMSD between the two chains.
    """
    chain_coords = np.array([residue["CA"].get_coord() for residue in chain_residues])
    new_chain_coords = np.array([residue["CA"].get_coord() for residue in new_chain_residues])
    new_chain_coords_aligned = kabsch_alignment(chain_coords, new_chain_coords)
    rmsd = np.sqrt(np.mean(np.sum((chain_coords - new_chain_coords_aligned) ** 2, axis=1)))
    return rmsd

def find_cath_chain_matches(args):
    """Find matching chains within CATH superfamilies."""
    # cath_domains = parse_cath_file(os.path.join(args.output_dir, "cath-domain-list-sample.txt"))
    cath_domains = parse_cath_file(os.path.join(args.output_dir, "cath-domain-list.txt"))
    dataset = CathDataset(os.path.join(args.output_dir, "pdb"), cath_domains)
    matches = []

    already_compared = set()
    with tqdm(total=len(cath_domains)) as pbar:
        for domain in cath_domains:
            pbar.update(1)
            domain_matches = []
            superfamily_domains = dataset.get_superfamily_domains(domain)

            if len(superfamily_domains) == 0:
                continue
            
            # try:
            domain_structure = dataset.parser.get_structure(domain.domain_name, 
                                                            dataset.get_domain_path(domain.domain_name))
            domain_chain = domain_structure[0][domain.domain_name[4]]
            domain_coords, domain_seq, domain_res_ids = dataset.extract_seq_and_res_ids(domain_chain)
            
            # Compare to random sample of 20 other domains to avoid N^2 comparisons
            for other_domain in np.random.choice(superfamily_domains, 20):
                print(other_domain)
                if other_domain.domain_name == domain.domain_name:
                    continue
                if set([domain.domain_name, other_domain.domain_name]) in already_compared:
                    continue
                already_compared.add((domain.domain_name, other_domain.domain_name))

                try:
                    other_structure = dataset.parser.get_structure(other_domain.domain_name,
                                                            dataset.get_domain_path(other_domain.domain_name))
                except FileNotFoundError:
                    continue
                other_chain = other_structure[0][other_domain.domain_name[4]]
                other_coords, other_seq, other_res_ids = dataset.extract_seq_and_res_ids(other_chain)

                aligned_res_ids1, aligned_res_ids2, aligned_seq1, aligned_seq2 = \
                    align_sequences_with_res_ids(domain_seq, torch.tensor(domain_res_ids),
                                                other_seq, torch.tensor(other_res_ids))

                seq_identity = compute_windowed_sequence_identity(aligned_seq1, aligned_seq2, 
                                                                window_size=args.motif_length)

                high_identity_regions = torch.where(seq_identity >= args.sequence_identity_threshold / 100)[0]
                
                if len(high_identity_regions) > 0:
                    for i in high_identity_regions:
                        start_res1 = aligned_res_ids1[i]
                        end_res1 = aligned_res_ids1[i+args.motif_length-1]
                        start_res2 = aligned_res_ids2[i]
                        end_res2 = aligned_res_ids2[i+args.motif_length-1]
                        if None in aligned_res_ids1[i:i+args.motif_length] or None in aligned_res_ids2[i:i+args.motif_length]:
                            # Skip if there are missing residues in the alignment, do not allow gaps
                            continue
                        chain_segment_residues = [residue for residue in domain_chain if residue.id[1] >= start_res1 and residue.id[1] <= end_res1]
                        other_chain_segment_residues = [residue for residue in other_chain if residue.id[1] >= start_res2 and residue.id[1] <= end_res2]
                        rmsd = calculate_rmsd(chain_segment_residues, other_chain_segment_residues)
                        superfamily = [domain.class_number, domain.architecture, domain.topology, domain.homologous_superfamily]
                        domain_matches.append({
                            "domain_name": domain.domain_name,
                            "matching_domain_name": other_domain.domain_name,
                            "identity": seq_identity[i].item(),
                            "overlap_residue_range_1": (start_res1, end_res1),
                            "overlap_residue_range_2": (start_res2, end_res2),
                            "rmsd": rmsd,
                            "superfamily": superfamily
                        })

                # except Exception as e:
                #     print(f"Error processing other domain {other_domain.domain_name}: {str(e)}")
                #     continue

            # except Exception as e:
            #     print(f"Error processing domain {domain.domain_name}: {str(e)}")
            #     continue
            if len(domain_matches) == 0:
                continue
            top_domain_match = sorted(domain_matches, key=lambda x: x["rmsd"])[0]
            matches.append(top_domain_match)

    output_file = os.path.join(args.output_dir, "cath_matches.json")
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=2)

    return matches
    

def select_top_case_studies(args):
    """Iterate over matches and identify the PDB chains  corresponding to the top 100 'top_linear_rmsd' values, excluding duplicates (i.e. if there is an entry for PDB A and PDB B, do not consider PDB B compared to PDB A). Also, only consider each domain once - i.e. if PDB B is a match for PDB A, do not search for additional matches to PDB B. Write these out to a json in the same format as the input."""
    with open(os.path.join(args.output_dir, "cath_matches.json"), "r") as f:
        matches = json.load(f)

    superfamily_top_case_studies = defaultdict(list)
    for domain_match in matches:
        superfamily_top_case_studies[tuple(domain_match["superfamily"])].append(domain_match)

    top_case_studies = []
    for superfamily, superfamily_matches in superfamily_top_case_studies.items():
        top_superfamily_matches = sorted(superfamily_matches, key=lambda x: x["rmsd"])
        top_case_studies.append(top_superfamily_matches[0])

    top_case_studies = sorted(top_case_studies, key=lambda x: x["rmsd"])[:args.num_top_case_studies]
    
    return top_case_studies


def inverse_fold_proteinmpnn(args):
    
    for top_match in os.listdir(os.path.join(args.output_dir, "top_case_studies")):
        pdb_id, chain_id = top_match.split("_")
        with open(os.path.join(args.output_dir, "top_case_studies", top_match, "match.json"), "r") as f:
            match = json.load(f)
        matching_pdb_id = match["matching_pdb_id"]
        matching_chain_id = match["matching_chain_id"]
        matching_pdb_path = os.path.join(args.output_dir, "top_case_studies", matching_pdb_id, matching_chain_id)
        matching_pdb = PDBParser().get_structure(matching_pdb_id, matching_pdb_path)
        matching_chain = matching_pdb[0][matching_chain_id]
        matching_chain_residues = list(matching_chain.get_residues())
        matching_chain_residues = matching_chain_residues[match["overlap_residue_range_2"][0]:match["overlap_residue_range_2"][1]+1]
        matching_chain_coords = np.array([residue["CA"].get_coord() for residue in matching_chain_residues])

        # Get the protein sequence
        matching_seq = get_sequence(matching_chain)
        matching_seq = matching_seq[match["overlap_residue_range_2"][0]:match["overlap_residue_range_2"][1]+1]

        # Get the fixed position mask
        fixed_position_mask = get_fixed_position_mask(matching_seq)

        # Get the ball mask
        ball_mask = get_ball_mask(matching_seq)

        # Get the protein
        protein = get_protein(matching_seq)

        # Load the model
        model = model_dict["proteinmpnn"]()
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "proteinmpnn.pt")))
        model.eval()

        # Inverse fold
        with torch.no_grad():
            pred = model.inverse_fold(protein, fixed_position_mask, ball_mask)

        # Save the inverse folded protein
        with open(os.path.join(args.output_dir, "top_case_studies", top_match, "inverse_folded_protein.pdb"), "w") as f:
            f.write(pred)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--motif_length", help="Length of the motif to use for matching", type=int, default=10)
    parser.add_argument("--sequence_identity_threshold", help="Sequence identity threshold for matching. E.g. if motif_length == 10, and sequence_identity_threshold == 90, then a match is found if >= 9/10 residues are identical.", type=int, default=90)
    parser.add_argument("--num_top_case_studies", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()

    # find_pdb_chain_matches()
    find_cath_chain_matches(args)
    top_case_studies = select_top_case_studies(args)

    for top_case_study in top_case_studies[:5]:
        print("Superfamily:", top_case_study["superfamily"])
        print("Domain:", top_case_study["domain_name"])
        print("Matching domain:", top_case_study["matching_domain_name"])
        print("Identity:", top_case_study["identity"])
        print("RMSD:", top_case_study["rmsd"])

    # inverse_fold_proteinmpnn()
    # inverse_fold_csdesign()
    # fold_esmfold()
    # calc_metrics()

# Example command:
# python -m src.bayes_design.experiments.scrmsd_experiment