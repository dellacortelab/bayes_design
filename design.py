import argparse
import torch
import numpy as np

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.utils import get_protein

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='bayes_struct')
parser.add_argument('--protein_id', help="The PDB id of the protein to redesign", default='6MRR')
parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='proximity')
parser.add_argument('--decode_algorithm', help="The algorithm used to decode masked parts of the sequence", choices=list(decode_algorithm_dict.keys()), default='beam')
parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
parser.add_argument('--n_beams', help="The number of beams, if using beam search decoding", type=int, default=16)
subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")
experiment_parser = subparsers.add_parser('experiment')
experiment_parser.add_argument('--name', help='The name of the experiment to run')


def example_design(args):
    
    device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")

    prob_model = model_dict[args.model_name](device=device)

    # Get sequence and structure of protein to redesign
    seq, struct = get_protein(args.protein_id)

    # Masked positions are the positions to predict/design
    # Default to predict all positions
    masked_positions = np.ones(len(seq))
    # Preserve fixed positions
    for i in range(0, len(args.fixed_positions), 2):
        # -1 because residues are 1-indexed
        fixed_range_start = args.fixed_positions[i] - 1
        # -1 because residues are 1-indexed and +1 because we are including the endpoint
        fixed_range_end = args.fixed_positions[i+1]
        masked_positions[fixed_range_start:fixed_range_end] = 0.
    masked_seq = ''.join(['-' if mask else char for char, mask in zip(seq, masked_positions)])

    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)
    
    # The decoding algorithm determines how the sequence is decoded
    if args.n_beams is None:
        designed_seq = decode_algorithm_dict[args.decode_algorithm](prob_model=prob_model, struct=struct, seq=masked_seq, decode_order=decode_order)
    else:
        designed_seq = decode_algorithm_dict[args.decode_algorithm](prob_model=prob_model, struct=struct, seq=masked_seq, decode_order=decode_order, n_beams=args.n_beams)

    return seq, masked_seq, designed_seq


if __name__ == '__main__':
    args = parser.parse_args()
    
    seqs = example_design(args)
    print(seqs)

