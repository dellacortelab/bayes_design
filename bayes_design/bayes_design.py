import argparse

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', help="The model to use for protein sequence design",
    choices=['bayes_struct', 'protein_mpnn', 'xlnet'], default='bayes_struct')
parser.add_argument('--protein_id', help="The PDB id of the protein to redesign", default='6MRR')
parser.add_argument('--decode_order', help="The order to decode masked parts of the \
    sequence", choices=['proximity', 'reverse_proximity', 'random', 'n_to_c'], default='proximity')
parser.add_argument('--decode_algorithm', help="The algorithm used to decode masked parts of the \
    sequence", choices=['greedy', 'beam', 'sample', 'random', 'compare', 'plot'], default='beam')

experiment_parser = parser.add_parser('experiment')
experiment_parser.add_argument('--name', help='The name of the experiment to run')



def example_design(args):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    prob_model = model_dict[args.model_name](device=device)

    # Get sequence and structure of protein to redesign
    seq, struct = get_protein(args.protein_id)

    # Masked positions are the positions to predict/design
    # Default to predict all positions
    masked_positions = np.ones(len(seq))
    # Preserve masked positions
    for i in range(len(args.masked_positions), step=2):
        mask_range_start = args.masked_positions[i]
        mask_range_end = args.masked_positions[i+1]
        masked_positions[mask_range_start:mask_range_end] = 0.
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


