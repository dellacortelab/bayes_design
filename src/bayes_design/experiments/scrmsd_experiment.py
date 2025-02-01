# scRMSD vs. ProteinMPNN in silico experiment

from Bio.PDB import PDBList, PDBParser
import torch
import numpy as np
import os
import ast
import argparse
from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.utils import get_protein, align_and_crop, get_ball_mask, get_fixed_position_mask
from bayes_design.experiments.cath import parse_cath_file
import logging
import bdb
from functools import lru_cache

from tqdm import tqdm

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import seq1 as three_letter_to_one_letter
from Bio.PDB.Polypeptide import PPBuilder
import json
import gzip
import shutil
import random

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

def ungzip_all_files_in_dir(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                output_file_path = os.path.splitext(gz_file_path)[0]  # Remove .gz extension
                
                if not os.path.exists(output_file_path):
                    # Un-gzip the file
                    with gzip.open(gz_file_path, 'rb') as gz_file, open(output_file_path, 'wb') as out_file:
                        shutil.copyfileobj(gz_file, out_file)
                    
                    # Don't remove the original .gz file - it allows rsync to skip it when downloading the original PDBs

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
def align_sequences_with_res_ids(seq1, res_ids1, seq2, res_ids2, coords1, coords2):
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
    alignments = pairwise2.align.globalxs(seq1, seq2, -1, -1, penalize_end_gaps=False)
    best_alignment = alignments[0]  # Use the best alignment

    aligned_seq1, aligned_seq2 = best_alignment[0], best_alignment[1]
    
    # Create mappings of aligned positions to res_ids
    aligned_res_ids1 = []
    aligned_res_ids2 = []
    aligned_coords1 = []
    aligned_coords2 = []    
    idx1, idx2 = 0, 0
    for i in range(len(aligned_seq1)):
        if aligned_seq1[i] != '-':
            aligned_res_ids1.append(res_ids1[idx1])
            aligned_coords1.append(coords1[idx1])
            idx1 += 1
        else:
            aligned_res_ids1.append(None)
            aligned_coords1.append(None)

        if aligned_seq2[i] != '-':
            aligned_res_ids2.append(res_ids2[idx2])
            aligned_coords2.append(coords2[idx2])
            idx2 += 1
        else:
            aligned_res_ids2.append(None)
            aligned_coords2.append(None)

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
    aligned_coords1 = aligned_coords1[beginning_idx:end_idx+1]
    aligned_coords2 = aligned_coords2[beginning_idx:end_idx+1]
    
    return aligned_res_ids1, aligned_res_ids2, aligned_seq1, aligned_seq2, aligned_coords1, aligned_coords2


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


def compute_full_sequence_identity(seq1, seq2):
    """Compute the sequence identity between two sequences. It is assumed that the sequences are
    the product of a sequence alignment and thus are of the same length
    """
    match_started = False
    cnt = 0
    match_cnt = 0
    for i, (char_a, char_b) in enumerate(zip(seq1, seq2)):
        if char_a != char_b or char_a == "-" or char_b == "-":
            if not match_started:
                continue
        else:
            if not match_started:
                match_started = True
                beg = i

            match_cnt += 1

        cnt += 1
        
    # Compute length of mismatching tail and remove penalty
    for i, (char_a, char_b) in enumerate(zip(reversed(seq1), reversed(seq2))):
        if char_a == char_b:
            break
    cnt -= i
    end = beg + cnt

    sequence_identity = match_cnt / cnt
    
    return sequence_identity, beg, end


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



def kabsch_alignment(target_coords, mobile_coords, coords_to_apply=None):
    # Step 1: Input validation
    assert target_coords.shape == mobile_coords.shape, "Input coordinate arrays must have the same shape."
    assert target_coords.shape[1] == 3, "Coordinate arrays must have shape N x 3."
    
    # Step 2: Compute centroids of coords1 and coords2
    centroid1 = np.mean(target_coords, axis=0)
    centroid2 = np.mean(mobile_coords, axis=0)
    
    # Step 3: Subtract the centroids
    target_coords_centered = target_coords - centroid1
    mobile_coords_centered = mobile_coords - centroid2
    
    # Step 4: Calculate the covariance matrix
    covariance_matrix = np.dot(mobile_coords_centered.T, target_coords_centered)
    
    # Step 5: Perform SVD on the covariance matrix
    U, S, Vt = np.linalg.svd(covariance_matrix)
    
    # Step 6: Compute the rotation matrix R
    d = np.linalg.det(np.dot(U, Vt))
    rotation_matrix = np.dot(U, np.dot(np.diag([1, 1, d]), Vt))
    
    if coords_to_apply is not None: # Overwrite mobile_coords_centered with coords_to_apply
        mobile_coords_centered = coords_to_apply - centroid2

    # Step 7: Rotate the second set of coordinates (coords2)
    coords2_aligned = np.dot(mobile_coords_centered, rotation_matrix)
    
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

def setup_json_file(file_path):
    """Initialize the file with an opening bracket if the file is empty or doesn't exist
    Returns:
        continuing_existing_run (bool): True if continuing existing run
    """
    if os.path.exists(file_path):
        # Check if file is non-empty
        if os.path.getsize(file_path) > 0:
            return True
        
    # File doesn't exist or is empty, initialize it
    with open(file_path, 'w') as f:
        f.write('[')

    return False

def append_record(file_path, data, is_first=False):
    """
    Append a single record to the file without closing the JSON array.
    Args:
        file_path (str): Path to the JSON file
        data (dict): The data to append
        is_first (bool): Whether this is the first record (to handle commas correctly)
    """
    with open(file_path, 'a') as f:
        # Add newline for readability, then the JSON-encoded record
        f.write('\n' + json.dumps(data, indent=2))
        # Add comma after record if it's not the first one
        f.write(',')

def finalize_json_file(file_path):
    """Close the JSON array with a closing bracket"""
    breakpoint()
    # TODO: test this

    # Remove the last comma and finalize the JSON array with a closing bracket
    with open(file_path, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
        f.write(b'\n]')

def find_cath_chain_matches(args, logdir, n_comparisons_per_domain=20, n_regions=20):
    """Find matching chains within CATH superfamilies."""
    os.makedirs(args.output_dir, exist_ok=True)

    # # Download pdb
    # pdb_dir = os.path.join(args.input_dir, "pdb")
    # pdb_rsync_cmd = f"rsync -rlpt -v -z --port=33444 rsync.wwpdb.org::ftp/data/structures/divided/pdb/ {pdb_dir}"
    # os.system(pdb_rsync_cmd)
    # ungzip_all_files_in_dir(pdb_dir)

    # Download CATH domain list
    if not os.path.exists(os.path.join(args.input_dir, "cath-domain-list.txt")):
        import subprocess
        subprocess.run(["curl", "https://download.cathdb.info/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt", "-o", os.path.join(args.input_dir, "cath-domain-list.txt")])

    # Load already compared set if it exists
    already_compared_path = os.path.join(args.output_dir, "already-compared.txt")
    if os.path.exists(already_compared_path):
        with open(already_compared_path) as f:
            # each line looks like: ("domain_1_name", "domain_2_name"), interpret with ast
            already_compared = set([ast.literal_eval(line) for line in f])
    else:
        already_compared = set()

    cath_domains = parse_cath_file(os.path.join(args.input_dir, "cath-domain-list.txt"))
    dataset = CathDataset(os.path.join(args.input_dir, "pdb"), cath_domains)
    matches_file = os.path.join(args.output_dir, "cath_matches.txt")
    continuing_existing_run = setup_json_file(matches_file)

    domain_1_w_error = set()
    domains_w_error = set()
    with open(already_compared_path, 'a') as f:
        with tqdm(total=len(cath_domains)) as pbar:
            for i, domain in enumerate(cath_domains):
                try:
                    pbar.update(1)

                    if domain.domain_name in domains_w_error:
                        continue
                    domain_matches = []

                    superfamily_domains = dataset.get_superfamily_domains(domain)

                    if len(superfamily_domains) == 0:
                        continue
                
                    domain_structure = dataset.parser.get_structure(domain.domain_name, 
                                                                    dataset.get_domain_path(domain.domain_name))
                    domain_chain = domain_structure[0][domain.domain_name[4]]
                    domain_coords, domain_seq, domain_res_ids = dataset.extract_seq_and_res_ids(domain_chain)
                    # Compare to random sample of 20 other domains to avoid N^2 comparisons
                    for j, other_domain in enumerate(np.random.choice(superfamily_domains, n_comparisons_per_domain)):
                        print(other_domain)
                        if other_domain.domain_name == domain.domain_name:
                            continue
                        if (domain.domain_name, other_domain.domain_name) in already_compared or (other_domain.domain_name, domain.domain_name) in already_compared:
                            continue
                        if other_domain.domain_name in domains_w_error:
                            continue

                        already_compared.add((domain.domain_name, other_domain.domain_name))
                        already_compared.add((other_domain.domain_name, domain.domain_name))
                        
                        f.write(str((domain.domain_name, other_domain.domain_name)) + "\n")
                        f.write(str((other_domain.domain_name, domain.domain_name)) + "\n")
                        try:
                            other_structure = dataset.parser.get_structure(other_domain.domain_name,
                                                                    dataset.get_domain_path(other_domain.domain_name))
                            
                            other_chain = other_structure[0][other_domain.domain_name[4]]
                            other_coords, other_seq, other_res_ids = dataset.extract_seq_and_res_ids(other_chain)

                            aligned_res_ids1, aligned_res_ids2, aligned_seq1, aligned_seq2 = \
                                align_sequences_with_res_ids(domain_seq, torch.tensor(domain_res_ids),
                                                            other_seq, torch.tensor(other_res_ids))

                            # Condition 1: sequence must have identiy >= full_sequence_identity_threshold
                            full_seq_identity, beg, end = compute_full_sequence_identity(aligned_seq1, aligned_seq2)
                            
                            if full_seq_identity < (args.full_sequence_identity_threshold / 100):
                                break

                            # Condition 2: sequence must have at least 1 motif with identity >= motif_sequence_identity_threshold
                            seq_identity = compute_windowed_sequence_identity(aligned_seq1, aligned_seq2, 
                                                                            window_size=args.motif_length)
                            high_identity_regions = torch.where(seq_identity >= args.motif_sequence_identity_threshold / 100)[0]
                            if len(high_identity_regions) == 0:
                                break

                            print("FOUND MATCH:", domain.domain_name, other_domain.domain_name)
                            # Sample n_regions indicies from high_identity_regions
                            idx = random.sample(range(len(high_identity_regions)), min(n_regions, len(high_identity_regions)))
                            idx = torch.tensor(idx)
                            sampled_high_identity_regions = high_identity_regions[idx]
                            for i in sampled_high_identity_regions:
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
                                    "beginning_idx": beg, # 0-indexed
                                    "end_idx": end, # 0-indexed, exclusive, such that included sequence is seq[beginning_idx:end_idx]
                                    "rmsd": rmsd,
                                    "superfamily": superfamily
                                })

                        except (KeyboardInterrupt, SystemExit, bdb.BdbQuit) as e:
                            raise e
                        except Exception as e:
                            domains_w_error.add(other_domain.domain_name)
                            msg = "Error message:" + "\n" + str(e) + "\n" + \
                                f"Error processing other domain {other_domain.domain_name}."
                            logging.error(msg)
                            continue

                except (KeyboardInterrupt, SystemExit, bdb.BdbQuit) as e:
                    raise e
                except Exception as e:
                    domains_w_error.add(domain.domain_name)
                    domain_1_w_error.add(domain.domain_name)
                    msg = "Error message:" + "\n" + str(e) + "\n" + \
                        f"Error processing other domain {domain.domain_name}." + "\n" + \
                        f"First domain error rate {len(domain_1_w_error) / i}."
                    logging.error(msg)
                    continue
                    
                if len(domain_matches) == 0:
                    continue
                top_domain_match = sorted(domain_matches, key=lambda x: x["rmsd"])[-1]

                # Add match
                append_record(matches_file, top_domain_match)

    finalize_json_file(matches_file)
    
def check_clash(dataset, domain_1, domain_2, overlap_residue_range_1, overlap_residue_range_2, structure_getter):
    """Calculate the RMSD between two chains.
    Args:
        chain_residues (list of Bio.PDB.Residue): The residues of the first chain.
        new_chain_residues (list of Bio.PDB.Residue): The residues of the second chain.
    Returns:
        rmsd (float): The RMSD between the two chains.
    """

    print(f"Cache status: {structure_getter.cache_info()}")

    # Load both proteins
    domain_1_structure = structure_getter(domain_1)
    domain_1_chain = domain_1_structure[0][domain_1[4]]
    domain_1_coords, domain_1_seq, domain_1_res_ids = dataset.extract_seq_and_res_ids(domain_1_chain)

    domain_2_structure = structure_getter(domain_2)
    domain_2_chain = domain_2_structure[0][domain_2[4]]
    domain_2_coords, domain_2_seq, domain_2_res_ids = dataset.extract_seq_and_res_ids(domain_2_chain)

    # Align res_id's, sequences, and coordinates (in sequence space)
    aligned_res_ids_1, aligned_res_ids_2, aligned_seq_1, aligned_seq_2, aligned_domain_1_coords, aligned_domain_2_coords = align_sequences_with_res_ids(
        domain_1_seq, torch.tensor(domain_1_res_ids), domain_2_seq, torch.tensor(domain_2_res_ids), domain_1_coords, domain_2_coords
    )

    # Remove positions that are None in either protein
    aligned_domain_coords = [(coord_1, coord_2, res_id_1, res_id_2) for coord_1, coord_2, res_id_1, res_id_2 in zip(aligned_domain_1_coords, aligned_domain_2_coords, aligned_res_ids_1, aligned_res_ids_2) if coord_1 is not None and coord_2 is not None]
    aligned_domain_1_coords, aligned_domain_2_coords, aligned_res_ids_1, aligned_res_ids_2 = zip(*aligned_domain_coords)

    full_seq_identity, beg, end = compute_full_sequence_identity(aligned_seq_1, aligned_seq_2)
    # Get coordinates corresponding to aligned region


    # Align the full proteins based on the rotottranslation that aligns the scaffolds
    aligned_domain_1_coords_scaffold = np.array([coord for coord, res_id in zip(aligned_domain_1_coords, aligned_res_ids_1) if not(res_id >= overlap_residue_range_1[0] and res_id <= overlap_residue_range_1[1])])
    aligned_domain_2_coords_scaffold = np.array([coord for coord, res_id in zip(aligned_domain_2_coords, aligned_res_ids_2) if not(res_id >= overlap_residue_range_2[0] and res_id <= overlap_residue_range_2[1])])
    aligned_domain_2_coords = kabsch_alignment(np.array(aligned_domain_1_coords_scaffold), np.array(aligned_domain_2_coords_scaffold), coords_to_apply=np.array(aligned_domain_2_coords))
    # Reset aligned_domain_2_coords_scaffold based on newly aligned coordinates
    aligned_domain_2_coords_scaffold = np.array([coord for coord, res_id in zip(aligned_domain_2_coords, aligned_res_ids_2) if not(res_id >= overlap_residue_range_2[0] and res_id <= overlap_residue_range_2[1])])

    if len(aligned_domain_1_coords_scaffold) < 20 or len(aligned_domain_2_coords_scaffold) < 20: # This filtering scheme assumes a reasonably large scaffold
        return True, None

    aligned_domain_1_coords_motif = np.array([coord for coord, res_id in zip(aligned_domain_1_coords, aligned_res_ids_1) if (res_id >= overlap_residue_range_1[0] and res_id <= overlap_residue_range_1[1])])
    aligned_domain_2_coords_motif = np.array([coord for coord, res_id in zip(aligned_domain_2_coords, aligned_res_ids_2) if (res_id >= overlap_residue_range_2[0] and res_id <= overlap_residue_range_2[1])])

    # Compute pairwise distances between aligned_domain_1_coords_motif and aligned_domain_2_coords_scaffold
    distances_1 = np.sqrt(((aligned_domain_1_coords_motif[:, None, :] - aligned_domain_2_coords_scaffold[None, :, :])**2).sum(axis=-1))
    if np.any(distances_1 < 2): # Give a lenient definition of a clash
        print(f"Clash found between {domain_1}, {domain_2}!")
        return True, None
    distances_2 = np.sqrt(((aligned_domain_2_coords_motif[:, None, :] - aligned_domain_1_coords_scaffold[None, :, :])**2).sum(axis=-1))
    if np.any(distances_2 < 2): # Give a lenient definition of a clash
        print(f"Clash found between {domain_1}, {domain_2}!")
        return True, None
    
    # Remove cases where scaffold A at position i is close to scaffold B at position j but scaffold B at position i is not close to scaffold B at position J
    
    # Calculate RMSD
    motif_rmsd = np.sqrt(((aligned_domain_1_coords_motif - aligned_domain_2_coords_motif)**2).sum(axis=-1)).mean().item()
            
    return False, motif_rmsd


def select_top_case_studies(args):
    """Iterate over matches and identify the PDB chains  corresponding to the top 100 'top_linear_rmsd' values, excluding duplicates (i.e. if there is an entry for PDB A and PDB B, do not consider PDB B compared to PDB A). Also, only consider each domain once - i.e. if PDB B is a match for PDB A, do not search for additional matches to PDB B. Write these out to a json in the same format as the input."""
    with open(os.path.join(args.output_dir, "cath_matches.txt"), "r") as f:
        matches = json.load(f)

    cath_domains = parse_cath_file(os.path.join(args.input_dir, "cath-domain-list.txt"))
    dataset = CathDataset(os.path.join(args.input_dir, "pdb"), cath_domains)
    
    superfamily_top_case_studies = defaultdict(list)
    n = 0

    @lru_cache(maxsize=700)
    def cache_structure_getter(domain):
        return dataset.parser.get_structure(domain, dataset.get_domain_path(domain))
    
    matches = matches[:100]
    with tqdm(total=len(matches)) as pbar:
        for domain_match in matches:
            pbar.update(1)
            if domain_match["domain_name"][:4] == domain_match["matching_domain_name"][:4]:
                continue
            # Check if matching motif from protein A clashes with a non-motif region in protein B. If so, drop it.
            clash, rmsd = check_clash(dataset, domain_match["domain_name"], domain_match["matching_domain_name"], domain_match["overlap_residue_range_1"], domain_match["overlap_residue_range_2"], cache_structure_getter)

            if clash:
                continue
            
            domain_match["rmsd"] = rmsd # Replace previous rmsd (motif-aligned motif rmsd) with new rmsd (scaffold-aligned motif rmsd)
            superfamily_top_case_studies[tuple(domain_match["superfamily"])].append(domain_match)
            n += 1
        print("N left after filtering clashes and same source pdbs:", n)

    top_case_studies = []
    for superfamily, superfamily_matches in superfamily_top_case_studies.items():
        top_superfamily_matches = sorted(superfamily_matches, key=lambda x: x["rmsd"])
        top_case_studies.append(top_superfamily_matches[0])

    top_case_studies = sorted(top_case_studies, key=lambda x: x["rmsd"])[-args.num_top_case_studies:]
    
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
    parser.add_argument("--input_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--full_sequence_identity_threshold", type=int, default=90)
    parser.add_argument("--motif_length", help="Length of the motif to use for matching", type=int, default=10)
    parser.add_argument("--motif_sequence_identity_threshold", help="Sequence identity threshold for matching. E.g. if motif_length == 10, and sequence_identity_threshold == 90, then a match is found if >= 9/10 residues are identical.", type=int, default=100)
    parser.add_argument("--num_top_case_studies", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()

    # Set random seed to avoid selecting different pairs for comparison each time.
    np.random.seed(0)
    random.seed(0)

    from datetime import datetime
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

    logdir = os.path.join(args.output_dir, "logs")
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, timestamp + ".txt")
    # Configure logging
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,      # Logging level
        format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
    )

    # find_cath_chain_matches(args, logdir)
    top_case_studies = select_top_case_studies(args)

    for top_case_study in top_case_studies[:5]:
        print("Superfamily:", top_case_study["superfamily"])
        print("Domain:", top_case_study["domain_name"])
        print("Overlap residue range:", top_case_study["overlap_residue_range_1"])
        print("Overlap residue range:", top_case_study["overlap_residue_range_2"])
        print("Matching domain:", top_case_study["matching_domain_name"])
        print("Identity:", top_case_study["identity"])
        print("RMSD:", top_case_study["rmsd"])

    # inverse_fold_proteinmpnn()
    # inverse_fold_csdesign()
    # fold_esmfold()
    # calc_metrics()

# Example command:
# python -m src.bayes_design.experiments.scrmsd_experiment