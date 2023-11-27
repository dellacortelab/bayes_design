import argparse
import torch
import pickle as pkl
import numpy as np
import random
import os

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.utils import get_protein, get_fixed_position_mask

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=list(model_dict.keys()), default='bayes_design')
parser.add_argument('--protein_id', help="The PDB id of the protein to redesign", default='6MRR')
parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=list(decode_order_dict.keys()), default='n_to_c')
parser.add_argument('--decode_algorithm', help="The algorithm used to decode masked parts of the sequence", choices=list(decode_algorithm_dict.keys()), default='beam')
parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces. Example: 3 10 14 14 17 20", nargs='*', type=int, default=[])
parser.add_argument('--n_beams', help="The number of beams, if using beam search decoding", type=int, default=16)
parser.add_argument('--redesign', help="Whether to redesign an existing sequence, using the existing sequence as bidirectional context. Default is to design from scratch.", action="store_true")
parser.add_argument('--device', help="The GPU index to use", type=int, default=0)
parser.add_argument('--bayes_balance_factor', help='A balancing factor to avoid a high probability ratio in the tails of the distribution. Suggested value: 0.002', default=0.002, type=float)
parser.add_argument('--temperature', help='The temperature to use for sampling', default=1.0, type=float)
parser.add_argument('--n_designs', help='The number of designs to generate', default=1, type=int)
parser.add_argument('--seed', help='The random seed to use', default=0, type=int)
parser.add_argument('--results_dir', help='The directory to save results to', default='./results')
parser.add_argument('--exclude_aa', nargs='+', default=[])

subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")
experiment_parser = subparsers.add_parser('experiment')
experiment_parser.add_argument('--name', help='The name of the experiment to run')


def example_design(args):

    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available()) else "cpu")
    
    if args.model_name == 'bayes_design':
        prob_model = model_dict[args.model_name](device=device, bayes_balance_factor=args.bayes_balance_factor)
    else:
        prob_model = model_dict[args.model_name](device=device)

    # Get sequence and structure of protein to redesign
    seq, struct = get_protein(args.protein_id)
    orig_seq = seq

    fixed_position_mask = get_fixed_position_mask(fixed_position_list=args.fixed_positions, seq_len=len(seq))
    masked_seq = ''.join(['-' if not fixed else char for char, fixed in zip(seq, fixed_position_mask)])

    # Decode order defines the order in which the masked positions are predicted
    decode_order = decode_order_dict[args.decode_order](masked_seq)
    
    if args.redesign:
        pass
    else:
        seq = masked_seq
    
    from_scratch = not args.redesign
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.decode_algorithm in ['max_prob_decode', 'combinations']:
        designed_seqs = decode_algorithm_dict[args.decode_algorithm](prob_model=prob_model, struct=struct, seq=seq, unmasked_seq=orig_seq, decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=from_scratch, temperature=args.temperature, n_beams=args.n_beams, exclude_aa=args.exclude_aa)
    else:
        designed_seqs = []
        for i in range(args.n_designs):
            designed_seq = decode_algorithm_dict[args.decode_algorithm](prob_model=prob_model, struct=struct, seq=seq, unmasked_seq=orig_seq, decode_order=decode_order, fixed_position_mask=fixed_position_mask, from_scratch=from_scratch, temperature=args.temperature, n_beams=args.n_beams, exclude_aa=args.exclude_aa)
            designed_seqs.append(designed_seq)

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f'{args.model_name}_{args.protein_id}_sequences.txt'), 'w') as f:
        for seq in designed_seqs:
            f.write(seq + '\n')

    return {"Original sequence":orig_seq, "Masked sequence (tokens to predict are indicated by a dash)":masked_seq, "Designed sequence":designed_seqs}


if __name__ == '__main__':
    args = parser.parse_args()
    
    seqs = example_design(args)
    if args.n_designs > 1:
        print(f"Designs written to {args.results_dir}/{args.model_name}_{args.protein_id}_sequences.txt")
    else:
        for k, v in seqs.items():
            print(f"{k}: {v}")

