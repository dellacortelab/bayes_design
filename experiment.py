#########################################################################
# Driver for experiments. Example commands are in experiments.md
#########################################################################

import argparse
import torch
import numpy as np

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.evaluate import metric_dict
from bayes_design.model import model_dict
from bayes_design.experiments import compare_seq_metric, compare_struct_correlation, viz_probs, make_pssm, make_hist, seq_filter

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")

compare_seq_probs_parser = subparsers.add_parser('compare_seq_metric')
compare_seq_probs_parser.add_argument('--sequences', help='String representations of protein sequences (required if --sequences_path is not provided)', nargs='+')
compare_seq_probs_parser.add_argument('--sequences_path', help='Path to a sequence file in fasta format (required if --sequences is not provided)')
compare_seq_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_seq_probs_parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='bayes_design')
compare_seq_probs_parser.add_argument('--metric', help="The metric with which to evaluate the sequence", choices=list(metric_dict.keys()), default='log_prob')
compare_seq_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
compare_seq_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
compare_seq_probs_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
compare_seq_probs_parser.add_argument('--redesign', help="Whether to redesign an existing sequence, using the existing sequence as bidirectional context. Default is to design from scratch.", action="store_true")
compare_seq_probs_parser.add_argument('--pssm_path', help='The path to a PSSM file used to score a sequence', default=None)
compare_seq_probs_parser.set_defaults(func=compare_seq_metric)

compare_struct_correlation_parser = subparsers.add_parser('compare_struct_correlation')
compare_struct_correlation_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_struct_correlation_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
compare_struct_correlation_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
compare_struct_correlation_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
compare_struct_correlation_parser.add_argument('--num_variants', help='The number of variants to generate for each sequence', default=10, type=int)
compare_struct_correlation_parser.add_argument('--perc_residues_to_mutate', help='The percentage of residues to mutate in each variant', default=0.1, type=float)
compare_struct_correlation_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
compare_struct_correlation_parser.add_argument('--device', help="The GPU index to use", type=int, default=0)
compare_struct_correlation_parser.set_defaults(func=compare_struct_correlation)

# compare_struct_probs_parser = subparsers.add_parser('compare_struct_probs')
# compare_struct_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
# compare_struct_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
# compare_struct_probs_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
# compare_struct_probs_parser.add_argument('--device', help="The GPU index to use", type=int, default=0)
# compare_struct_probs_parser.set_defaults(func=compare_struct_probs)

viz_probs_parser = subparsers.add_parser('viz_probs')
viz_probs_parser.add_argument('--sequence', help='String representations of the protein sequence to redesign', required=True)
viz_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
viz_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
viz_probs_parser.add_argument('--from_scratch', help="Whether to design a new sequence from scratch. Default is to condition on existing sequence from provided PDB file.", action="store_true")
viz_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
viz_probs_parser.add_argument('--results_path', help="The directory to store results", default='./results/probs_viz.png')
viz_probs_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
viz_probs_parser.set_defaults(func=viz_probs)


pssm_parser = subparsers.add_parser('make_pssm')
pssm_parser.add_argument('--sequences', help='String representations of the protein sequence to redesign', nargs='+', default=None)
pssm_parser.add_argument('--sequences_path', help='Path to the sequence file containing >1 sequence', required=True)
pssm_parser.add_argument('--pssm_path', help='Path to the pssm file', required=True)
pssm_parser.set_defaults(func=make_pssm)

hist_parser = subparsers.add_parser('make_hist')
hist_parser.add_argument('--sequences', help='String representations of the protein sequence to redesign', nargs='+', default=None)
hist_parser.add_argument('--sequences_path', help='Path to the sequence file containing >1 sequence', default=None)
hist_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
hist_parser.add_argument('--model_name', help="The name of the model to use", choices=list(model_dict.keys()), default='pssm')
hist_parser.add_argument('--metric', help="The metric with which to evaluate the sequence", choices=list(metric_dict.keys()), default='log_prob')
hist_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
hist_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
hist_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
hist_parser.add_argument('--redesign', help="Whether to redesign the sequence", action="store_true")
hist_parser.add_argument('--pssm_path', help='Path to the pssm file', default=None)
hist_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
hist_parser.set_defaults(func=make_hist)

# Takes the same arguments as make_hist
seq_filter_parser = subparsers.add_parser('seq_filter')
seq_filter_parser.add_argument('--sequences', help='String representations of the protein sequence to redesign', nargs='+', default=None)
seq_filter_parser.add_argument('--sequences_path', help='Path to the sequence file containing >1 sequence', default=None)
seq_filter_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
seq_filter_parser.add_argument('--model_name', help="The name of the model to use", choices=list(model_dict.keys()), default='pssm')
seq_filter_parser.add_argument('--metric', help="The metric with which to evaluate the sequence", choices=list(metric_dict.keys()), default='log_prob')
seq_filter_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
seq_filter_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
seq_filter_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
seq_filter_parser.add_argument('--redesign', help="Whether to redesign the sequence", action="store_true")
seq_filter_parser.add_argument('--pssm_path', help='Path to the pssm file', default=None)
seq_filter_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
seq_filter_parser.add_argument('--n_seqs', help="The number of sequences to select", default=10, type=int)
seq_filter_parser.set_defaults(func=seq_filter)

# viz_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
# viz_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
# viz_probs_parser.add_argument('--from_scratch', help="Whether to design a new sequence from scratch. Default is to condition on existing sequence from provided PDB file.", action="store_true")
# viz_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
# viz_probs_parser.add_argument('--results_path', help="The directory to store results", default='./results/probs_viz.png')
# viz_probs_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
# Whether to construct a PSSM from a list of sequences, or score a set of sequences under a pssm

def parse_seq(args):
    """Parse the sequence from the appropriate command line argument"""
    if args.sequences is not None:
        return args
    elif args.sequences_path is not None:
        args.sequences = []
        with open(args.sequences_path, 'r') as f:
            for line in f:
                if line[0] == '>':
                    continue
                seq = line.strip()
                args.sequences.append(seq)
    return args

if __name__ == '__main__':
    args = parser.parse_args()
    args = parse_seq(args)
    args.func(args)