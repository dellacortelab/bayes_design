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
from scipy import stats
from typing import List, Dict, Tuple
import pandas as pd

from tqdm import tqdm

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqUtils import seq1 as three_letter_to_one_letter
from Bio.PDB.Polypeptide import PPBuilder
import json
import gzip
import shutil
import random
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

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
    
    def extract_seq_and_res_ids(self, chain, atoms_to_return=["CA"]):
        sequence = []
        res_ids = []
        coords = []
        for residue in chain:
            if residue.has_id("CA"):  # Filter to only include amino acids
                sequence.append(three_letter_to_one_letter(residue.get_resname()))
                res_ids.append(residue.id[1])
                try:
                    coords.append([residue[atom].get_coord() for atom in atoms_to_return])
                except KeyError:
                    coords.append([[None] * 3 for atom in atoms_to_return])
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
    A, D = len(coords1[0]), len(coords1[0][0])
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
            aligned_coords1.append([[None] * D] * A)

        if aligned_seq2[i] != '-':
            aligned_res_ids2.append(res_ids2[idx2])
            aligned_coords2.append(coords2[idx2])
            idx2 += 1
        else:
            aligned_res_ids2.append(None)
            aligned_coords2.append([[None] * D] * A)

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
    try:
        # Step 5: Perform SVD on the covariance matrix
        U, S, Vt = np.linalg.svd(covariance_matrix)
    except:
        breakpoint()
    
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
    # Collapse atom dimension
    aligned_domain_1_coords, aligned_domain_2_coords = [coord[0] for coord in aligned_domain_1_coords], [coord[0] for coord in aligned_domain_2_coords]

    if len(aligned_res_ids_1) > 650:
        return True, None
    
    def check_proximity_to_disordered(aligned_res_ids, overlap_residue_range):
        # Return early if motif is within 5 positions of n terminus, c terminus, or a missing residue
        n_terminus_res_id = [res_id for res_id in aligned_res_ids if res_id is not None][0]
        distance_to_n_terminus = overlap_residue_range[0] - n_terminus_res_id
        if distance_to_n_terminus <= 5:
            print("CLOSE TO N")
            return True
        # Check proximity to c terminus
        c_terminus_res_id = [res_id for res_id in aligned_res_ids if res_id is not None][-1]
        distance_to_c_terminus = c_terminus_res_id - overlap_residue_range[1]
        if distance_to_c_terminus <= 5:
            print("CLOSE TO C")
            return True
        # Check proximity to missing residue
        overlap_indices = [i for i, res_id in enumerate(aligned_res_ids) if res_id in overlap_residue_range]
        none_indices = [i for i, res_id in enumerate(aligned_res_ids) if res_id is None]
        for overlap_idx in overlap_indices:
            for none_idx in none_indices:
                distance = abs(overlap_idx - none_idx)
                if distance <= 5:
                    print("CLOSE TO NONE")
                    return True
                
        return False
            
    if check_proximity_to_disordered(aligned_res_ids_1, overlap_residue_range_1):
        return True, None
        
    if check_proximity_to_disordered(aligned_res_ids_2, overlap_residue_range_2):
        return True, None

    # Remove positions that are None in either protein
    aligned_domain_coords = [(coord_1, coord_2, res_id_1, res_id_2) for coord_1, coord_2, res_id_1, res_id_2 in zip(aligned_domain_1_coords, aligned_domain_2_coords, aligned_res_ids_1, aligned_res_ids_2) if coord_1[0] is not None and coord_2[0] is not None]
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
    
    # We want the scaffolds to be reasonably well-aligned
    scaffold_rmsd = np.sqrt(((aligned_domain_1_coords_scaffold - aligned_domain_2_coords_scaffold)**2).sum(axis=-1).mean()).item()
    if scaffold_rmsd > 6:
        print("SCAFFOLD error too large")
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

    # Calculate RMSD
    motif_rmsd = np.sqrt(((aligned_domain_1_coords_motif - aligned_domain_2_coords_motif)**2).sum(axis=-1).mean()).item()
            
    return False, motif_rmsd


def filter_matches(args):
    """Iterate over matches and identify the PDB chains  corresponding to the top 100 'top_linear_rmsd' values, excluding duplicates (i.e. if there is an entry for PDB A and PDB B, do not consider PDB B compared to PDB A). Also, only consider each domain once - i.e. if PDB B is a match for PDB A, do not search for additional matches to PDB B. Write these out to a json in the same format as the input."""
    with open(os.path.join(args.output_dir, "cath_matches.txt"), "r") as f:
        matches = json.load(f)

    cath_domains = parse_cath_file(os.path.join(args.input_dir, "cath-domain-list.txt"))
    dataset = CathDataset(os.path.join(args.input_dir, "pdb"), cath_domains)
    matches_file = os.path.join(args.output_dir, "cath_matches_remove_clashes.txt")
    existing_run = setup_json_file(matches_file)
    if existing_run: raise ValueError("File cath_matches_remove_clashes.txt already exists")
    
    n = 0

    @lru_cache(maxsize=700)
    def cache_structure_getter(domain):
        return dataset.parser.get_structure(domain, dataset.get_domain_path(domain))
    
    # matches = matches[:500]
    with tqdm(total=len(matches)) as pbar:
        for domain_match in matches:
            pbar.update(1)
            # Remove matches within the same pdb id, as these are less interesting
            if domain_match["domain_name"][:4] == domain_match["matching_domain_name"][:4]:
                continue
            # Check if matching motif from protein A clashes with a non-motif region in protein B. If so, drop it.
            clash, rmsd = check_clash(dataset, domain_match["domain_name"], domain_match["matching_domain_name"], domain_match["overlap_residue_range_1"], domain_match["overlap_residue_range_2"], cache_structure_getter)
            if clash:
                continue
            
            domain_match["rmsd"] = rmsd # Replace previous rmsd (motif-aligned motif rmsd) with new rmsd (scaffold-aligned motif rmsd)
            # Add match
            append_record(matches_file, domain_match)
            n += 1
        print("N left after filtering clashes and same source pdbs:", n)
        
    finalize_json_file(matches_file)

def select_top_case_studies(args):
    """Iterate over matches and identify the PDB chains  corresponding to the top 100 'top_linear_rmsd' values, excluding duplicates (i.e. if there is an entry for PDB A and PDB B, do not consider PDB B compared to PDB A). Also, only consider each domain once - i.e. if PDB B is a match for PDB A, do not search for additional matches to PDB B. Write these out to a json in the same format as the input."""
    with open(os.path.join(args.output_dir, "cath_matches_remove_clashes.txt"), "r") as f:
        matches = json.load(f)
    
    superfamily_top_case_studies = defaultdict(list)
    # matches = matches[:500]

    # Drop duplicates
    print("Number of matches before deduplication on pdb id:", len(matches))
    matches_dedup = []
    domain_name_to_idx = dict()
    for domain_match in matches:
        domain_name = domain_match["domain_name"][:4]
        matching_domain_name = domain_match["matching_domain_name"][:4]
        
        domain_name_present = domain_name in domain_name_to_idx.keys()
        matching_domain_name_present = matching_domain_name in domain_name_to_idx.keys()
        names_to_drop = []
        if domain_name_present and matching_domain_name_present:
            if matches_dedup[domain_name_to_idx[domain_name]]["rmsd"] < domain_match["rmsd"] and matches_dedup[domain_name_to_idx[matching_domain_name]]["rmsd"] < domain_match["rmsd"]:
                domain_name_idx = domain_name_to_idx[domain_name]
                matching_domain_name_idx = domain_name_to_idx[matching_domain_name]
                names_to_drop.append(matches_dedup[domain_name_idx]["domain_name"][:4])
                names_to_drop.append(matches_dedup[domain_name_idx]["matching_domain_name"][:4])
                names_to_drop.append(matches_dedup[matching_domain_name_idx]["domain_name"][:4])
                names_to_drop.append(matches_dedup[matching_domain_name_idx]["matching_domain_name"][:4])
                for name in names_to_drop:
                    domain_name_to_idx.pop(name, None)
                matches_dedup[domain_name_idx] = None
                matches_dedup[matching_domain_name_idx] = None
                
                matches_dedup.append(domain_match)
                domain_name_to_idx[domain_name] = len(matches_dedup) - 1
                domain_name_to_idx[matching_domain_name] = len(matches_dedup) - 1
            else: # do not add
                pass 
        elif domain_name_present:
            if matches_dedup[domain_name_to_idx[domain_name]]["rmsd"] < domain_match["rmsd"]:  # replace
                domain_name_idx = domain_name_to_idx[domain_name]
                names_to_drop.append(matches_dedup[domain_name_idx]["domain_name"][:4])
                names_to_drop.append(matches_dedup[domain_name_idx]["matching_domain_name"][:4])
                for name in names_to_drop:
                    domain_name_to_idx.pop(name, None)
                matches_dedup[domain_name_idx] = None
                
                matches_dedup.append(domain_match)
                domain_name_to_idx[domain_name] = len(matches_dedup) - 1
                domain_name_to_idx[matching_domain_name] = len(matches_dedup) - 1
            else: # do not add
                pass 
        elif matching_domain_name_present:
            if matches_dedup[domain_name_to_idx[matching_domain_name]]["rmsd"] < domain_match["rmsd"]:  # replace
                matching_domain_name_idx = domain_name_to_idx[matching_domain_name]
                names_to_drop.append(matches_dedup[matching_domain_name_idx]["domain_name"][:4])
                names_to_drop.append(matches_dedup[matching_domain_name_idx]["matching_domain_name"][:4])
                for name in names_to_drop:
                    domain_name_to_idx.pop(name, None)
                matches_dedup[matching_domain_name_idx] = None

                matches_dedup.append(domain_match)
                domain_name_to_idx[domain_name] = len(matches_dedup) - 1
                domain_name_to_idx[matching_domain_name] = len(matches_dedup) - 1
                
        else: # add
            matches_dedup.append(domain_match)
            domain_name_to_idx[domain_name] = len(matches_dedup) - 1
            domain_name_to_idx[matching_domain_name] = len(matches_dedup) - 1

    matches_dedup = [match for match in matches_dedup if match is not None]
    print("Number of matches after deduplication on pdb id:", len(matches_dedup))
    assert len(set([domain_match["domain_name"][:4] for domain_match in matches_dedup])) == len(matches_dedup)
    assert len(set([domain_match["matching_domain_name"][:4] for domain_match in matches_dedup])) == len(matches_dedup)

    # Sort by superfamily
    for domain_match in matches_dedup:
        superfamily_top_case_studies[tuple(domain_match["superfamily"])].append(domain_match)

    top_case_studies = []
    for superfamily, superfamily_matches in superfamily_top_case_studies.items():
        top_superfamily_matches = sorted(superfamily_matches, key=lambda x: x["rmsd"])
        top_case_studies.append(top_superfamily_matches[-1])

    top_case_studies = sorted(top_case_studies, key=lambda x: x["rmsd"])[-args.num_top_case_studies:]

    with open(os.path.join(args.output_dir, "cath_matches_sorted.txt"), "w") as f:
        json.dump(top_case_studies, f, indent=2)

    return top_case_studies

def graft_sequence(seq_1, seq_2, seq_3):
    seq_1_grafted = []
    for seq_1_char, seq_2_char, seq_3_char in zip(list(seq_1), list(seq_2), list(seq_3)):
        if seq_1_char != "-":
            seq_1_grafted.append(seq_1_char)
        elif seq_2_char != "-":
            seq_1_grafted.append(seq_2_char)
        elif seq_3_char != "-":
            seq_1_grafted.append(seq_3_char)
        else:
            raise ValueError("All sequences contain the '-' character at the same position")            

    return "".join(seq_1_grafted)

def inverse_fold(args):


    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available()) else "cpu")
    cs_design = model_dict["cs_design"](device=device)
    protein_mpnn = model_dict["protein_mpnn"](device=device)

    with open(os.path.join(args.output_dir, "cath_matches_sorted.txt"), "r") as f:
        matches = json.load(f)

    cath_domains = parse_cath_file(os.path.join(args.input_dir, "cath-domain-list.txt"))
    dataset = CathDataset(os.path.join(args.input_dir, "pdb"), cath_domains)

    
    results = []
    for match in matches:
        domain_1_name = match["domain_name"]
        domain_2_name = match["matching_domain_name"]

        domain_1_structure = dataset.parser.get_structure(domain_1_name, 
                                                        dataset.get_domain_path(domain_1_name))
        domain_1_chain = domain_1_structure[0][domain_1_name[4]]
        domain_1_coords, domain_1_seq, domain_1_res_ids = dataset.extract_seq_and_res_ids(domain_1_chain, atoms_to_return=["N", "CA", "C", "O"])

        domain_2_structure = dataset.parser.get_structure(domain_2_name, 
                                                        dataset.get_domain_path(domain_2_name))
        domain_2_chain = domain_2_structure[0][domain_2_name[4]]
        domain_2_coords, domain_2_seq, domain_2_res_ids = dataset.extract_seq_and_res_ids(domain_2_chain, atoms_to_return=["N", "CA", "C", "O"])

        aligned_res_ids1, aligned_res_ids2, aligned_seq1, aligned_seq2, aligned_coords1, aligned_coords2 = align_sequences_with_res_ids(
            domain_1_seq, torch.tensor(domain_1_res_ids),
            domain_2_seq, torch.tensor(domain_2_res_ids),
            domain_1_coords, domain_2_coords
        )
        
        aligned_coords1, aligned_coords2 = torch.tensor(np.array(aligned_coords1, dtype=np.float32)), torch.tensor(np.array(aligned_coords2, dtype=np.float32))

        # Get the fixed position mask
        fixed_position_mask = np.ones(len(aligned_res_ids1), dtype=bool)
        for i, domain_1_res_id in enumerate(aligned_res_ids1):
            if domain_1_res_id is not None and domain_1_res_id >= match["overlap_residue_range_1"][0] and domain_1_res_id <= match["overlap_residue_range_1"][1]:
                fixed_position_mask[i] = 0

        aligned_seq1_masked = ''.join(['-' if not fixed else char for char, fixed in zip(aligned_seq1, fixed_position_mask)])
        aligned_seq2_masked = ''.join(['-' if not fixed else char for char, fixed in zip(aligned_seq2, fixed_position_mask)])
        # Decode order defines the order in which the masked positions are predicted
        decode_order = decode_order_dict["n_to_c"](aligned_seq1)
        
        os.makedirs(os.path.join(args.output_dir, "reference_coords"), exist_ok=True)
        coords_1_path = os.path.join(args.output_dir, "reference_coords", domain_1_name + ".pt")
        coords_2_path = os.path.join(args.output_dir, "reference_coords", domain_2_name + ".pt")
        torch.save(aligned_coords1, coords_1_path)
        torch.save(aligned_coords2, coords_2_path)
        
        pred_sequence = decode_algorithm_dict["greedy"](prob_model=cs_design, struct=(aligned_coords1, aligned_coords2), seq=(aligned_seq1_masked, aligned_seq2_masked), decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=True)[0]
        pred_sequence = graft_sequence(pred_sequence, aligned_seq1, aligned_seq2)
        results.append({
            "model_name": "cs_design",
            "domain_name_pro": domain_1_name,
            "domain_name_anti": domain_2_name,
            "sequence_pro": aligned_seq1,
            "sequence_anti": aligned_seq2,
            "motif_mask": (~fixed_position_mask).astype(int).tolist(),
            "pred_sequence": pred_sequence,
            "coords_path_pro": coords_1_path,
            "coords_path_anti": coords_2_path,
        })
        
        pred_sequence = decode_algorithm_dict["greedy"](prob_model=cs_design, struct=(aligned_coords2, aligned_coords1), seq=(aligned_seq2_masked, aligned_seq1_masked), decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=True)[0]
        pred_sequence = graft_sequence(pred_sequence, aligned_seq2, aligned_seq1)
        results.append({
            "model_name": "cs_design",
            "domain_name_pro": domain_2_name,
            "domain_name_anti": domain_1_name,
            "sequence_pro": aligned_seq2,
            "sequence_anti": aligned_seq1,
            "motif_mask": (~fixed_position_mask).astype(int).tolist(),
            "pred_sequence": pred_sequence,
            "coords_path_pro": coords_2_path,
            "coords_path_anti": coords_1_path,
        })

        pred_sequence = decode_algorithm_dict["greedy"](prob_model=protein_mpnn, struct=aligned_coords1, seq=aligned_seq1_masked, decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=True)
        pred_sequence = graft_sequence(pred_sequence, aligned_seq1, aligned_seq2)
        results.append({
            "model_name": "protein_mpnn",
            "domain_name_pro": domain_1_name,
            "domain_name_anti": domain_2_name,
            "sequence_pro": aligned_seq1,
            "sequence_anti": aligned_seq2,
            "motif_mask": (~fixed_position_mask).astype(int).tolist(),
            "pred_sequence": pred_sequence,
            "coords_path_pro": coords_1_path,
            "coords_path_anti": coords_2_path,
        })

        pred_sequence = decode_algorithm_dict["greedy"](prob_model=protein_mpnn, struct=aligned_coords2, seq=aligned_seq2_masked, decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=True)
        pred_sequence = graft_sequence(pred_sequence, aligned_seq2, aligned_seq1)
        results.append({
            "model_name": "protein_mpnn",
            "domain_name_pro": domain_2_name,
            "domain_name_anti": domain_1_name,
            "sequence_pro": aligned_seq2,
            "sequence_anti": aligned_seq1,
            "motif_mask": (~fixed_position_mask).astype(int).tolist(),
            "pred_sequence": pred_sequence,
            "coords_path_pro": coords_2_path,
            "coords_path_anti": coords_1_path,
        })
        

    with open(os.path.join(args.output_dir, "sequence_predictions.txt"), "w") as f:
        json.dump(results, f, indent=2)



def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def esmfold(args):

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model.esm = model.esm.half() # This is okay, it was trained in fp16
    model = model.to("cuda:1")

    with open(os.path.join(args.output_dir, "sequence_predictions.txt"), "r") as f:
        sequence_predictions = json.load(f)

    pred_dir = os.path.join(args.output_dir, "pred_coords")
    os.makedirs(pred_dir, exist_ok=True)

    with tqdm(total=len(sequence_predictions)) as pbar:
        for sequence_prediction in sequence_predictions:
            pbar.update(1)
            # # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
            # # model.trunk.set_chunk_size(64)
            # # Length: 700. Fits on a 24GB VRAM GPU
            # test_protein = "MGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLFMGAGASAEEKHSRELEKKLKEDAEKDARTVKLLLLGAGESGKSTIVKQMKIIHQDGYSLEECLEFIAIIYGNTLQSILAIVRAMTTLNIQYGDSARQDDARKLMHMADTIEEGTMPKEMSDIIQRLWKDSGIQACFERASEYQLNDSAGYYLSDLERLVTPGYVPTEQDVLRSRVKTTGIIETQFSFKDLNFRMFDVGGQRSERKKWIHCFEGVTCIIFIAALSAYDMVLVEDDEVNRMHESLHLFNSICNHRYFATTSIVLFLNKKDVFFEKIKKAHLSICFPDYDGPNTYEDAGNYIKVQFLELNMRRDVKEIYSHMTCATDTQNVKFVFDAVTDIIIKENLKDCGLF"
            
            tokenized_input = tokenizer([sequence_prediction["pred_sequence"]], return_tensors="pt", add_special_tokens=False)['input_ids']
            tokenized_input = tokenized_input.to("cuda:1")
            
            with torch.no_grad():
                output = model(tokenized_input)

            # Save pred coords
            pred_coords = output["positions"][-1, 0, :, :4, :].detach().cpu()
            domain_name_pro, domain_name_anti, model_name = sequence_prediction["domain_name_pro"], sequence_prediction["domain_name_anti"], sequence_prediction["model_name"]
            file_name = f"pro_{domain_name_pro}_anti_{domain_name_anti}_model_{model_name}"
            coords_file_name = file_name + ".pt"
            pred_coords_path = os.path.join(pred_dir, coords_file_name)
            torch.save(pred_coords, pred_coords_path)
            sequence_prediction["pred_coords_path"] = pred_coords_path

            # Save pdb
            pdb_file_name = file_name + ".pdb"
            pdb_path = os.path.join(pred_dir, pdb_file_name)
            pdb = convert_outputs_to_pdb(output)
            with open(pdb_path, "w") as f:
                f.writelines(pdb)
            sequence_prediction["pdb_file_name"] = pdb_path


    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        json.dump(sequence_predictions, f, indent=2)
    

def compute_rmsd(coords1: torch.Tensor, coords2: torch.Tensor, mask: List[int] = None) -> float:
    """
    Compute RMSD between two sets of coordinates.
    
    Args:
        coords1: N x 4 x 3 tensor of coordinates
        coords2: N x 4 x 3 tensor of coordinates
        mask: Optional list of 0s and 1s for masking specific positions
    
    Returns:
        float: RMSD value
    """
    # Align full proteins on scaffold CA
    motif_mask = torch.tensor(mask).bool()
    nan_position_mask = torch.isnan(coords1.sum(-1)) | torch.isnan(coords2.sum(-1))
    coords1, coords2, motif_mask = coords1[~nan_position_mask], coords2[~nan_position_mask], motif_mask[~nan_position_mask]
    coords2 = torch.tensor(kabsch_alignment(target_coords=coords1[~motif_mask].numpy(), mobile_coords=coords2[~motif_mask].numpy(), coords_to_apply=coords2.numpy()))

    coords1 = coords1[motif_mask]
    coords2 = coords2[motif_mask]
    
    diff = coords1 - coords2
    squared_diff = torch.sum(diff * diff, dim=1)
    rmsd = torch.sqrt(torch.mean(squared_diff)).item()
    return rmsd

def analyze_predictions(predictions: List[Dict]) -> pd.DataFrame:
    """
    Analyze predictions and create a DataFrame with RMSD values.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        pd.DataFrame: Analysis results
    """
    results = []
    fail_cases = []
    i = 0
    for pred in predictions:
        i += 1
        coords_path_pro = pred["coords_path_pro"]
        coords_path_anti = pred["coords_path_anti"]
        pred_coords_path = pred["pred_coords_path"]
        motif_mask = pred["motif_mask"]
        model_name = pred["model_name"]
        domain_pro = pred["domain_name_pro"]
        domain_anti = pred["domain_name_anti"]
        
        coords_pro = torch.load(coords_path_pro)[:, 1]
        coords_anti = torch.load(coords_path_anti)[:, 1]
        pred_coords = torch.load(pred_coords_path)[:, 1] # N x 4 x 3 torch.tensor
        
        # Compute RMSDs
        rmsd_pro = compute_rmsd(pred_coords, coords_pro, motif_mask)
        rmsd_anti = compute_rmsd(pred_coords, coords_anti, motif_mask)
        
        results.append({
            'model_name': model_name,
            'domain_pro': domain_pro,
            'domain_anti': domain_anti,
            'rmsd_pro': rmsd_pro,
            'rmsd_anti': rmsd_anti,
            'rmsd_diff': rmsd_anti - rmsd_pro  # Positive means prefers pro conformation
        })
    
    return pd.DataFrame(results)

def analyze_conformational_preference(df: pd.DataFrame, model: str) -> Tuple[float, float]:
    """
    Analyze whether a model shows statistically significant conformational preference.
    
    Args:
        df: DataFrame with analysis results
        model: Model name to analyze
    
    Returns:
        Tuple[float, float]: T-statistic and p-value
    """
    model_data = df[df['model_name'] == model]['rmsd_diff']
    t_stat, p_value = stats.ttest_1samp(model_data, popmean=0)
    return t_stat, p_value

def compare_models(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compare performance between models.
    
    Args:
        df: DataFrame with analysis results
    
    Returns:
        Tuple[float, float]: T-statistic and p-value
    """
    csdesign_data = df[df['model_name'] == 'cs_design']['rmsd_diff']
    proteinmpnn_data = df[df['model_name'] == 'protein_mpnn']['rmsd_diff']
    
    # Perform paired t-test on the RMSD differences
    t_stat, p_value = stats.ttest_rel(
        csdesign_data, 
        proteinmpnn_data
    )
    return t_stat, p_value


def dual_conformation_analysis(df):
    """
    Compare algorithms' ability to design for both conformations of the same protein.
    Pairs up predictions where domain_name_pro for one is domain_name_anti for the other.
    
    Returns:
    dict: Statistical analysis of dual conformation performance
    """
    # Create dictionary to store paired designs for each algorithm
    csdesign_pairs = {}
    proteinmpnn_pairs = {}
    
    # Group designs by algorithm
    csdesign_data = df[df['model_name'] == 'cs_design']
    proteinmpnn_data = df[df['model_name'] == 'protein_mpnn']
    
    # Function to process each algorithm's data
    def pair_designs(data, pairs_dict):
        for _, row in data.iterrows():
            domain_key = tuple(sorted([row['domain_pro'], row['domain_anti']]))
            if domain_key not in pairs_dict:
                pairs_dict[domain_key] = {'conf_a': None, 'conf_b': None, 'indices': []}
            
            pairs_dict[domain_key]['indices'].append(row.name)

            # Determine which conformation this design represents
            if pairs_dict[domain_key]['conf_a'] is None:
                pairs_dict[domain_key]['conf_a'] = row['rmsd_pro']
            else:
                pairs_dict[domain_key]['conf_b'] = row['rmsd_pro']
    
    # Pair up the designs for each algorithm
    pair_designs(csdesign_data, csdesign_pairs)
    pair_designs(proteinmpnn_data, proteinmpnn_pairs)

    df["summed_rmsd"] = None
    # Update summed_rmsd in dataframe
    for pairs_dict in [csdesign_pairs, proteinmpnn_pairs]:
        for domain_key, pair_data in pairs_dict.items():
            if pair_data['conf_a'] is not None and pair_data['conf_b'] is not None:
                summed_rmsd = pair_data['conf_a'] + pair_data['conf_b']
                # Update both entries in the pair with the summed RMSD
                for idx in pair_data['indices']:
                    df.at[idx, 'summed_rmsd'] = summed_rmsd
    
    # Calculate combined RMSD scores for each algorithm
    csdesign_combined = []
    proteinmpnn_combined = []
    
    for domain_key in csdesign_pairs.keys():
        if (csdesign_pairs[domain_key]['conf_a'] is not None and 
            csdesign_pairs[domain_key]['conf_b'] is not None and
            domain_key in proteinmpnn_pairs and
            proteinmpnn_pairs[domain_key]['conf_a'] is not None and
            proteinmpnn_pairs[domain_key]['conf_b'] is not None):
            
            cs_score = (csdesign_pairs[domain_key]['conf_a'] + 
                        csdesign_pairs[domain_key]['conf_b'])
            mpnn_score = (proteinmpnn_pairs[domain_key]['conf_a'] + 
                        proteinmpnn_pairs[domain_key]['conf_b'])
            
            csdesign_combined.append(cs_score)
            proteinmpnn_combined.append(mpnn_score)
    
    # Perform paired t-test on combined scores
    t_stat, p_value = stats.ttest_rel(
        csdesign_combined,
        proteinmpnn_combined
    )
    
    return df, {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'csdesign_mean_combined': np.mean(csdesign_combined),
        'proteinmpnn_mean_combined': np.mean(proteinmpnn_combined),
        'n_pairs': len(csdesign_combined)
    }

def run_analysis(predictions: List[Dict]) -> Dict:
    """
    Run complete statistical analysis.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Dict: Analysis results
    """
    # Create DataFrame
    df = analyze_predictions(predictions)
    
    # Analyze each model's conformational preference
    csdesign_t, csdesign_p = analyze_conformational_preference(df, 'cs_design')
    proteinmpnn_t, proteinmpnn_p = analyze_conformational_preference(df, 'protein_mpnn')
    
    # Compare models
    model_comp_t, model_comp_p = compare_models(df)
    
    # Calculate effect sizes
    csdesign_effect = df[df['model_name'] == 'cs_design']['rmsd_diff'].mean()
    proteinmpnn_effect = df[df['model_name'] == 'protein_mpnn']['rmsd_diff'].mean()

    df, dual_conformation_result = dual_conformation_analysis(df)
    notable_cases = find_notable_cases(df)
    
    return {
        'CSDesign': {
            't_statistic': csdesign_t,
            'p_value': csdesign_p,
            'effect_size': csdesign_effect
        },
        'ProteinMPNN': {
            't_statistic': proteinmpnn_t,
            'p_value': proteinmpnn_p,
            'effect_size': proteinmpnn_effect
        },
        'Model_Comparison': {
            't_statistic': model_comp_t,
            'p_value': model_comp_p,
            'effect_size': csdesign_effect - proteinmpnn_effect
        },
        'Model_Comparison_Dual_Conformation': dual_conformation_result,
        **notable_cases
    }


def print_analysis_results(results: Dict):
    """
    Print analysis results in a readable format.
    
    Args:
        results: Dictionary of analysis results
    """
    print("Statistical Analysis Results\n")
    
    print("CSDesign Conformational Preference:")
    print(f"t-statistic: {results['CSDesign']['t_statistic']:.3f}")
    print(f"p-value: {results['CSDesign']['p_value']:.3e}")
    print(f"Effect size (mean RMSD difference): {results['CSDesign']['effect_size']:.3f} Å\n")
    
    print("ProteinMPNN Conformational Preference:")
    print(f"t-statistic: {results['ProteinMPNN']['t_statistic']:.3f}")
    print(f"p-value: {results['ProteinMPNN']['p_value']:.3e}")
    print(f"Effect size (mean RMSD difference): {results['ProteinMPNN']['effect_size']:.3f} Å\n")
    
    print("Model Comparison (CSDesign vs ProteinMPNN):")
    print(f"t-statistic: {results['Model_Comparison']['t_statistic']:.3f}")
    print(f"p-value: {results['Model_Comparison']['p_value']:.3e}")
    print(f"Effect size (difference in mean RMSD differences): {results['Model_Comparison']['effect_size']:.3f} Å")

    print("Model Comparison (CSDesign vs ProteinMPNN):")
    print(f"\nDual Conformation Analysis:")
    print(f"Number of paired instances: {results['Model_Comparison_Dual_Conformation']['n_pairs']}")
    print(f"t-statistic: {results['Model_Comparison_Dual_Conformation']['t_statistic']:.4f}")
    print(f"p-value: {results['Model_Comparison_Dual_Conformation']['p_value']:.4f}")
    print(f"CSDesign mean combined RMSD: {results['Model_Comparison_Dual_Conformation']['csdesign_mean_combined']:.4f}")
    print(f"ProteinMPNN mean combined RMSD: {results['Model_Comparison_Dual_Conformation']['proteinmpnn_mean_combined']:.4f}")

def find_notable_cases(df):
    """
    Find and return notable cases from the dataset
    """
    results = defaultdict(dict)
    
    # 1. Best individual design (smallest RMSD_pro)
    df_protein_mpnn = df[df["model_name"] == "protein_mpnn"]
    df_cs_design = df[df["model_name"] == "cs_design"]

    best_design_protein_mpnn = df_protein_mpnn.loc[df_protein_mpnn['rmsd_pro'].idxmin()]
    results["protein_mpnn"]['best_rmsd_pro'] = {
        'model_name': best_design_protein_mpnn['model_name'],
        'domain_pro': best_design_protein_mpnn['domain_pro'],
        'domain_anti': best_design_protein_mpnn['domain_anti'],
        'rmsd_pro': best_design_protein_mpnn['rmsd_pro']
    }

    best_design_cs_design = df_cs_design.loc[df_cs_design['rmsd_pro'].idxmin()]
    results["cs_design"]['best_rmsd_pro'] = {
        'model_name': best_design_cs_design['model_name'],
        'domain_pro': best_design_cs_design['domain_pro'],
        'domain_anti': best_design_cs_design['domain_anti'],
        'rmsd_pro': best_design_cs_design['rmsd_pro']
    }
    
    # 2. Best preference (largest RMSD_diff)
    df_cs_design['rmsd_diff'] = df_cs_design['rmsd_anti'] - df_cs_design['rmsd_pro']
    best_pref_cs_design = df_cs_design.loc[df_cs_design['rmsd_diff'].idxmax()]  # minimum because smaller RMSD_pro is better
    results["cs_design"]['best_preference'] = {
        'model_name': best_pref_cs_design['model_name'],
        'domain_pro': best_pref_cs_design['domain_pro'],
        'domain_anti': best_pref_cs_design['domain_anti'],
        'rmsd_diff': best_pref_cs_design['rmsd_diff']
    }

    df_protein_mpnn['rmsd_diff'] = df_protein_mpnn['rmsd_anti'] - df_protein_mpnn['rmsd_pro']
    best_pref_protein_mpnn = df_protein_mpnn.loc[df_protein_mpnn['rmsd_diff'].idxmax()]  # minimum because smaller RMSD_pro is better
    results["protein_mpnn"]['best_preference'] = {
        'model_name': best_pref_protein_mpnn['model_name'],
        'domain_pro': best_pref_protein_mpnn['domain_pro'],
        'domain_anti': best_pref_protein_mpnn['domain_anti'],
        'rmsd_diff': best_pref_protein_mpnn['rmsd_diff']
    }
    
    # 3. Best overall design (smallest summed RMSD)
    best_sum_cs_design = df_cs_design.loc[df_cs_design['summed_rmsd'].idxmin()]
    results["cs_design"]['best_total_rmsd'] = {
        'model_name': best_sum_cs_design['model_name'],
        'domain_pro': best_sum_cs_design['domain_pro'],
        'domain_anti': best_sum_cs_design['domain_anti'],
        'summed_rmsd': best_sum_cs_design['summed_rmsd']
    }
    best_sum_protein_mpnn = df_protein_mpnn.loc[df_protein_mpnn['summed_rmsd'].idxmin()]
    results["protein_mpnn"]['best_total_rmsd'] = {
        'model_name': best_sum_protein_mpnn['model_name'],
        'domain_pro': best_sum_protein_mpnn['domain_pro'],
        'domain_anti': best_sum_protein_mpnn['domain_anti'],
        'summed_rmsd': best_sum_protein_mpnn['summed_rmsd']
    }
    
    # 4. Largest difference between algorithms
    # First, create a DataFrame with paired designs
    
    paired_data = []
    for pro_domain in df['domain_pro'].unique():
        cs_design = df[(df['domain_pro'] == pro_domain) & 
                           (df['model_name'] == 'cs_design')]
        prot_mpnn = df[(df['domain_pro'] == pro_domain) & 
                           (df['model_name'] == 'protein_mpnn')]
        
        if not cs_design.empty and not prot_mpnn.empty:
            cs_diff = cs_design['rmsd_diff'].iloc[0]
            mpnn_diff = prot_mpnn['rmsd_diff'].iloc[0]
            algo_diff = cs_diff - mpnn_diff # Higher numbers favor cs_design

            cs_summed_rmsd = cs_design["summed_rmsd"].iloc[0]
            prot_summed_rmsd = prot_mpnn["summed_rmsd"].iloc[0]
            summed_rmsd_diff = prot_summed_rmsd - cs_summed_rmsd # Higher numbers favor cs_design
            
            paired_data.append({
                'domain_pro': pro_domain,
                'domain_anti_cs': cs_design['domain_anti'].iloc[0],
                'domain_anti_mpnn': prot_mpnn['domain_anti'].iloc[0],
                'diff_between_algorithm_anti_pro_diffs': algo_diff,
                'cs_diff': cs_diff,
                'mpnn_diff': mpnn_diff,
                'diff_between_algorithm_summed_diffs': summed_rmsd_diff,
                'cs_design_summed_rmsd': cs_summed_rmsd,
                'protein_mpnn_summed_rmsd': prot_summed_rmsd,
            })
    
    paired_df = pd.DataFrame(paired_data)
    if not paired_df.empty:
        largest_algo_diff = paired_df.loc[paired_df['diff_between_algorithm_anti_pro_diffs'].idxmax()]
        results['cs_design']['largest_algorithm_difference'] = largest_algo_diff.to_dict()
        largest_algo_diff = paired_df.loc[paired_df['diff_between_algorithm_anti_pro_diffs'].idxmin()]
        results['protein_mpnn']['largest_algorithm_difference'] = largest_algo_diff.to_dict()
        
        largest_algo_diff = paired_df.loc[paired_df['diff_between_algorithm_summed_diffs'].idxmax()]
        results['cs_design']['largest_algorithm_summed_difference'] = largest_algo_diff.to_dict()
        largest_algo_diff = paired_df.loc[paired_df['diff_between_algorithm_summed_diffs'].idxmin()]
        results['protein_mpnn']['largest_algorithm_summed_difference'] = largest_algo_diff.to_dict()
    
    return results

def print_notable_cases(notable_cases):
    """
    Print notable cases in a readable format
    """
    
    for model_name in ["cs_design", "protein_mpnn"]:
        cases = notable_cases[model_name]
        print("\nNotable Cases:")
        print("\n1. Best Individual Design (Smallest RMSD_pro):")
        print(f"Model: {cases['best_rmsd_pro']['model_name']}")
        print(f"Domain Pro: {cases['best_rmsd_pro']['domain_pro']}")
        print(f"Domain Anti: {cases['best_rmsd_pro']['domain_anti']}")
        print(f"RMSD Pro: {cases['best_rmsd_pro']['rmsd_pro']:.3f}")
        
        print("\n2. Best Preference (Largest RMSD difference):")
        print(f"Model: {cases['best_preference']['model_name']}")
        print(f"Domain Pro: {cases['best_preference']['domain_pro']}")
        print(f"Domain Anti: {cases['best_preference']['domain_anti']}")
        print(f"RMSD Difference: {cases['best_preference']['rmsd_diff']:.3f}")
        
        print("\n3. Best Overall Design (Smallest summed RMSD):")
        print(f"Model: {cases['best_total_rmsd']['model_name']}")
        print(f"Domain Pro: {cases['best_total_rmsd']['domain_pro']}")
        print(f"Domain Anti: {cases['best_total_rmsd']['domain_anti']}")
        print(f"Total RMSD: {cases['best_total_rmsd']['summed_rmsd']:.3f}")
        
        print("\n4. Largest Algorithm Difference:")
        algo_diff = cases['largest_algorithm_difference']
        print(f"Domain Pro: {algo_diff['domain_pro']}")
        print(f"CSDesign Domain Anti: {algo_diff['domain_anti_cs']}")
        print(f"ProteinMPNN Domain Anti: {algo_diff['domain_anti_mpnn']}")
        print(f"CSDesign RMSD Difference: {algo_diff['cs_diff']:.3f}")
        print(f"ProteinMPNN RMSD Difference: {algo_diff['mpnn_diff']:.3f}")
        print(f"Absolute Difference between Algorithms: {algo_diff['diff_between_algorithm_summed_diffs']:.3f}")


        print("\n4. Largest Algorithm Difference:")
        algo_diff = cases['largest_algorithm_summed_difference']
        print(f"Domain Pro: {algo_diff['domain_pro']}")
        print(f"CSDesign Domain Anti: {algo_diff['domain_anti_cs']}")
        print(f"ProteinMPNN Domain Anti: {algo_diff['domain_anti_mpnn']}")
        print(f"CSDesign Summed RMSD: {algo_diff['cs_design_summed_rmsd']:.3f}")
        print(f"ProteinMPNN Summed RMSD: {algo_diff['protein_mpnn_summed_rmsd']:.3f}")
        print(f"Absolute Difference between Algorithms: {algo_diff['diff_between_algorithm_summed_diffs']:.3f}")

def compute_metrics(args):
    # Questions: 
    # For CSDesign, do "pro" designs prefer conformation a over conformation b with statistical significance?
    # For ProteinMPNN, same question
    # Does CSDesign or ProteinMPNN outperform the other with statistical significance?
    # Load your predictions
    with open(os.path.join(args.output_dir, "predictions.txt"), "r") as f:
        predictions = json.load(f)

    # Run analysis
    results = run_analysis(predictions)


    # Print results
    print_analysis_results(results)
    print_notable_cases(results)

# TODO: handle coords from align_sequnces_with_res_ids globally



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--full_sequence_identity_threshold", type=int, default=90)
    parser.add_argument("--motif_length", help="Length of the motif to use for matching", type=int, default=10)
    parser.add_argument("--motif_sequence_identity_threshold", help="Sequence identity threshold for matching. E.g. if motif_length == 10, and sequence_identity_threshold == 90, then a match is found if >= 9/10 residues are identical.", type=int, default=100)
    parser.add_argument("--num_top_case_studies", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    
    args = parser.parse_args()

    # Set random seed to avoid selecting different pairs for comparison each time.
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

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
    # TODO: Fix find_cath_chain_matches to use old syntax (no coords)
    # filter_matches(args)
    # top_case_studies = select_top_case_studies(args)
    # print(len(top_case_studies))

    # for top_case_study in top_case_studies[-5:]:
    #     print("Superfamily:", top_case_study["superfamily"])
    #     print("Domain:", top_case_study["domain_name"])
    #     print("Overlap residue range:", top_case_study["overlap_residue_range_1"])
    #     print("Overlap residue range:", top_case_study["overlap_residue_range_2"])
    #     print("Matching domain:", top_case_study["matching_domain_name"])
    #     print("Identity:", top_case_study["identity"])
    #     print("RMSD:", top_case_study["rmsd"])

    inverse_fold(args)
    esmfold(args)
    # compute_metrics(args)

# Example command:
# python -m src.bayes_design.experiments.scrmsd_experiment