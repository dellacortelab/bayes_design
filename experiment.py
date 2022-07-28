import argparse
import torch
import numpy as np

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.experiments import compare_seq_probs, compare_struct_probs

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")
compare_seq_probs_parser = subparsers.add_parser('compare_seq_probs')
compare_seq_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
compare_seq_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_seq_probs_parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='bayes_design')
compare_seq_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='proximity')
compare_seq_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
compare_seq_probs_parser.set_defaults(func=compare_seq_probs)

compare_seq_probs_parser = subparsers.add_parser('compare_struct_probs')
compare_seq_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
compare_seq_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_seq_probs_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
compare_seq_probs_parser.set_defaults(func=compare_struct_probs)

# Make log prob evaluation work in batch version
# Compare correlation of bayes_design with trRosetta and protein_mpnn with trRosetta

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)