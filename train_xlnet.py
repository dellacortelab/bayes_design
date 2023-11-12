
from bayes_design.protein_mpnn.training import main as run_training

import argparse

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data") 
argparser.add_argument("--path_for_outputs", type=str, default="./test", help="path for logs and model weights")
argparser.add_argument("--previous_checkpoint", type=str, default="", help="path for previous model weights, e.g. file.pt")
argparser.add_argument("--num_epochs", type=int, default=200, help="number of epochs to train for")
argparser.add_argument("--save_model_every_n_epochs", type=int, default=10, help="save model weights every n epochs")
argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2, help="reload training data every n epochs")
argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
argparser.add_argument("--batch_size", type=int, default=10000, help="number of tokens for one batch")
argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
argparser.add_argument("--hidden_dim", type=int, default=128, help="hidden model dimension")
argparser.add_argument("--num_encoder_layers", type=int, default=3, help="number of encoder layers") 
argparser.add_argument("--num_decoder_layers", type=int, default=3, help="number of decoder layers")
argparser.add_argument("--num_neighbors", type=int, default=48, help="number of neighbors for the sparse graph")   
argparser.add_argument("--dropout", type=float, default=0.1, help="dropout level; 0.0 means no dropout")
argparser.add_argument("--backbone_noise", type=float, default=0.2, help="amount of noise added to backbone during training")   
argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
argparser.add_argument("--gradient_norm", type=float, default=-1.0, help="clip gradient norm, set to negative to omit clipping")
argparser.add_argument("--mixed_precision", type=bool, default=True, help="train with mixed precision")
argparser.add_argument("--model_name", type=str, default="xlnet", choices=["xlnet", "protein_mpnn"], help="model name")

args = argparser.parse_args()    


if __name__ == "__main__":
    run_training(args, model_name='xlnet')