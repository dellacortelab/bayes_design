import argparse
import torch
import numpy as np

from bayes_design.decode import decode_order_dict, decode_algorithm_dict
from bayes_design.model import model_dict
from bayes_design.experiments import compare_seq_probs

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Whether to run an experiment instead of using the base design functionality")
compare_seq_probs_parser = subparsers.add_parser('compare_seq_probs')
compare_seq_probs_parser.add_argument('--sequences', help='String representations of protein sequences', nargs='+', required=True)
compare_seq_probs_parser.add_argument('--protein_id', help="The PDB id of the structure for the protein sequences", default='6MRR')
compare_seq_probs_parser.add_argument('--model_name', help="The model to use for protein sequence design", choices=['bayes_struct', 'protein_mpnn', 'protein_mpnn_batch', 'xlnet'], default='bayes_struct')
compare_seq_probs_parser.add_argument('--decode_order', help="The order to decode masked parts of the sequence", choices=['proximity', 'reverse_proximity', 'random', 'n_to_c'], default='proximity')
compare_seq_probs_parser.add_argument('--fixed_positions', help="The beginnings and ends of residue ranges (includes endpoints [], 1-indexed) to remain fixed and not predicted, separated by spaces", nargs='*', type=int, default=[])
compare_seq_probs_parser.set_defaults(func=compare_seq_probs)

# original sequence: KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
# python3 design.py --model_name bayes_struct --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 32 --fixed_positions 34 34 MLPEGWKLIKDPKTGQDMCFNEITKEITAQRPVG
# python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 32 --fixed_positions 34 34 KLPEGWVEVTDPKTGKKLYYNTKTKEITEEKPVG
# python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --decode_algorithm random --n_beams 32 --fixed_positions 34 34
# python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name bayes_struct --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG MLPEGWKLIKDPKTGQDMCFNEITKEITAQRPVG KLPEGWVEVTDPKTGKKLYYNTKTKEITEEKPVG
# python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG MLPEGWKLIKDPKTGQDMCFNEITKEITAQRPVG KLPEGWVEVTDPKTGKKLYYNTKTKEITEEKPVG
# Sequence source:                  orig        bayes_struct    protein_mpnn
# Scores under bayes_struct model:  -89.158     -32.946         -61.638
# Scores under protein_mpnn model:  -64.238     -44.980         -28.066

# python3 experiment.py compare_seq_probs --protein_id plastic_degrading_enzyme --model_name bayes_struct --decode_order n_to_c --fixed_positions 138 138 184 184 216 216 65 65 97 97 139 139 162 162 66 67 137 137 211 212 215 215 217 217 223 225 261 261 221 226 181 181 218 218 252 252 269 269 274 281 --sequences TDPGNGSGYQRGPDPTVSFLEAARGQYTVDTERVSSLVGGFGGGTIHYPEDVSGTMAAIVVIPGYVSAESSIEWWGPKLASYGFVVMTIDTNTGFDQPPSRATQINAALDYLVDQNSDNGSPVQGMIDTSRLGVIGWSMGGGGTIRVASQGRIKAAIPLAPWDTSSYYARRAEAATMIFACESDVVAPVGLHASPFYNALPSSIDKAFVEINNGSHFCANGGGINNDVLGRLGVSWMKLHLDEDGRYNQFLCGPNHESDFSISEYRGNCPYGS QIAPCGHKWVFGPEPTRENLKAPHGFWAVSQCEISASVEGFGGGTIHMPTNVKGRLPAVVIMHGYVSDKDSIAFWGPRLASFGFIVLVINWKSPDCQPEQMAQEIRAALDHMDQWNKNPKSPIHGMIDEKRLGVIGWSMGGGATIIVASDGMFKAAIPLCPWGPNTDPAKQAKADTLIFGCENDTVCPPEKHSRPMWDAVPKTVDRMFVEINDGSHFCWTGGGINNCVLRLLTTSWVRLHLMRDMQVEKFLCGPEIDNDPSISEFESNCPFGQ SVEPGGSPYVRGPEPTKALLAAPKGPWAVAEEEISASVEGFGGGTVYYPENVTGKLPAVVIIPGYVSSKESVAGWGPALASFGFVVYVIDWRSGDDQPAEVAEEIKAALDLLEEMNKDPNSPIKGLIDENRLGVIGWSMGGGATIIVASTGRVKAAIPLVPWLPSTEPAKKATANTLILACENDKVTPPEKYSKPAYEAIPKTIDRMLVLINNASHFCGAGGGINNPVLNLYVISWLRLHLQLDKRVEQFLCGPAITNDPSISEYRNNCPFGT

# Redesign sequence for stefan

# Compare correlation of bayes_design with trRosetta and protein_mpnn with trRosetta

if __name__ == '__main__':
    args = parser.parse_args()
    args.func(args)