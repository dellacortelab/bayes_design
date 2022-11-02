#########################################################################
# Driver for experiments. Example commands are in experiments.md
#########################################################################

import argparse
import torch
import numpy as np

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.evaluate import metric_dict
from bayes_design.model import model_dict
from bayes_design.experiments import compare_seq_metric, compare_struct_probs, viz_probs

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")

compare_seq_probs_parser = subparsers.add_parser('compare_seq_metric')
compare_seq_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
compare_seq_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_seq_probs_parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='bayes_design')
compare_seq_probs_parser.add_argument('--metric', help="The metric with which to evaluate the sequence", choices=list(metric_dict.keys()), default='log_prob')
compare_seq_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
compare_seq_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
compare_seq_probs_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0., type=float)
compare_seq_probs_parser.add_argument('--from_scratch', help="Whether to treat a sequence as designed from scratch when computing metrics. Default is to condition on existing sequence from provided PDB file.", action="store_true")
compare_seq_probs_parser.set_defaults(func=compare_seq_metric)

compare_struct_probs_parser = subparsers.add_parser('compare_struct_probs')
compare_struct_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
compare_struct_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_struct_probs_parser.add_argument('--results_dir', help="The directory to store results", default='./results')
compare_struct_probs_parser.set_defaults(func=compare_struct_probs)

viz_probs_parser = subparsers.add_parser('viz_probs')
viz_probs_parser.add_argument('--sequence', help='String representations of the protein sequence to redesign', required=True)
viz_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
viz_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
viz_probs_parser.add_argument('--from_scratch', help="Whether to design a new sequence from scratch. Default is to condition on existing sequence from provided PDB file.", action="store_true")
viz_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
viz_probs_parser.add_argument('--results_path', help="The directory to store results", default='./results/probs_viz.png')
viz_probs_parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0., type=float)
viz_probs_parser.set_defaults(func=viz_probs)

# Make log prob evaluation work in batch version
# Compare correlation of bayes_design with trRosetta and protein_mpnn with trRosetta

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)