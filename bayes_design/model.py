import re
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
import torch
import random
import os.path
from transformers import XLNetTokenizer, XLNetLMHeadModel
from protein_mpnn_utils import ProteinMPNN
from tr_rosetta_pytorch import trRosettaNetwork
from tr_rosetta_pytorch.utils import preprocess

from .utils import AMINO_ACID_ORDER, get_protein

class XLNetWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.tokenizer = XLNetTokenizer.from_pretrained('Rostlab/prot_xlnet', do_lower_case=False)
        self.model = XLNetLMHeadModel.from_pretrained('Rostlab/prot_xlnet')
        self.model.to(self.device)

        xlnet_vocab_dict = self.tokenizer.get_vocab()
        xlnet_vocab_dict['▁X'] = xlnet_vocab_dict['X']
        self.canonical_idx_to_xlnet_idx = torch.tensor([xlnet_vocab_dict['▁' + aa] for aa in AMINO_ACID_ORDER])

    def forward(self, seq, decode_order, token_to_decode):
        """Accept an amino acid sequence, return class probabilities for the next token
        Args:
            seq (len N list of len L_seq str): a string representation of an amino 
                acid sequence with unknown residues indicated with a dash (-)
            decode_order (len L list): list of the order of indices to decode.
                This determines the values in the permutation mask. Each index 
                attends to all indices that occur previously in the decoding_order.
            token_to_decode (int): index in the range [0, L-1] indicating which 
                token to decode next. If not specified, this function will predict the
                first token indicated with dash (-)
        Returns:
            probs ((21) torch.Tensor): a vector of probabilities for the next
                token
        """
        # Determine which indices should be masked for prediction
        masked_indices = [i for i, char in enumerate(seq) if char == '-']
        masked_indices.append(token_to_decode)
        
        # Replace rare amino acids with "X"
        seq = re.sub(r"[UZOB]", "X", seq)
        # Huggingface XLNet expects a space-separated sequence
        seq = " ".join(seq)
        # Replace '-' characters with the <mask> token
        seq = re.sub(r"-", "<mask>", seq)
        # Convert characters to token ids
        input_ids = torch.tensor(self.tokenizer.encode(seq)).unsqueeze(0).to(self.device)
        L = input_ids.shape[1]
        
        # Mask unknown tokens and the token to predict
        perm_mask = torch.ones((1, L, L), dtype=torch.float).to(self.device)
        decode_order = torch.tensor(decode_order)
        for i, idx in enumerate(decode_order):
            # idx attends to all tokens before idx in the decoding_order
            perm_mask[:, idx, decode_order[:i]] = 0.0

        # Indicate which token to decode
        target_mapping = torch.zeros((1, 1, L), dtype=torch.float).to(self.device)  # Shape [batch_size, 1, seq_length]
        target_mapping[0, 0, token_to_decode] = 1.0  # Our first prediction will be the indicated token

        # Get probabilities
        with torch.no_grad():
            output_dict = self.model(input_ids=input_ids, perm_mask=perm_mask, target_mapping=target_mapping, return_dict=True)
            logits = output_dict['logits']  # logits has shape [batch_size, 1, config.vocab_size]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            index_corrected_probs = probs[0][0][self.canonical_idx_to_xlnet_idx]

        return index_corrected_probs

class TransformerXLU100Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class ProGenWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
        
class ProteinMPNNWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        #v_48_010=version with 48 edges 0.10A noise
        model_name = "v_48_030"
        backbone_noise=0.00               # Standard deviation of Gaussian noise to add to backbone atoms
        path_to_model_weights='/content/ProteinMPNN/vanilla_proteinmpnn/vanilla_model_weights'          
        hidden_dim = 128
        num_layers = 3 
        model_folder_path = path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
        checkpoint_path = model_folder_path + f'{model_name}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device) 
        print('Number of edges:', checkpoint['num_edges'])
        noise_level_print = checkpoint['noise_level']
        print(f'Training noise level: {noise_level_print}A')
        self.model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded")
    
    def forward(self, seq, struct, decode_order, token_to_decode):
        """Accept an amino acid sequence and protein structure 
        coordinates, return class probabilities for the next token
        Args:
            seq (len L str): a string representation of an amino 
                acid sequence with unknown residues indicated with a dash (-)
            struct ((L x 4 x 3) torch.Tensor): batch_size x seq_length x 
                num_atoms x num_coordinates tensor
            decode_order (len L list): list of the order of indices to decode.
                This determines the values in the permutation mask. Each index 
                attends to all indices that occur previously in the decoding_order.
            token_to_decode (int): index in the range [0, L-1] indicating which 
                token to decode next. If not specified, this function will predict the
                first token indicated with a dash
        Returns:
            probs ((N x 21) torch.Tensor): a vector of probabilities for the next
                token
        """
        L = len(seq)

        # Convert amino acid character to index
        seq = re.sub(r"-", "X", seq)
        seq = torch.tensor([AMINO_ACID_ORDER.index(aa) for aa in seq]).to(self.device)
        with torch.no_grad():
            # Default values
            mask = torch.ones(1, L).float().to(self.device)
            chain_M = torch.ones(1, L).float().to(self.device)
            chain_M_pos = torch.ones(1, L).float().to(self.device)
            chain_encoding_all = torch.ones(1, L).float().to(self.device)
            residue_idx = torch.arange(L).unsqueeze(0).to(self.device)
            bias_AAs_np = np.zeros(len(AMINO_ACID_ORDER))
            bias_by_res = torch.zeros([1, L, 21]).to(self.device)
            # randn determines the decoding order. The indices in the randn vector with the lowest values will be decoded first
            randn = torch.argsort(torch.tensor(decode_order)).to(self.device)
            # Predict tokens except 'X'
            omit_AAs_np = np.array([AA in ['X'] for AA in AMINO_ACID_ORDER]).astype(np.float32)
            out = self.model.sample(X=struct.unsqueeze(0).to(self.device), randn=randn, S_true=seq.unsqueeze(0), chain_mask=chain_M, chain_encoding_all=chain_encoding_all, residue_idx=residue_idx, mask=mask, chain_M_pos=chain_M_pos, bias_AAs_np=bias_AAs_np, omit_AAs_np=omit_AAs_np, bias_by_res=bias_by_res)
            probs = out['probs']
        # Return the probabilities for th 0th chain and the 0th residue
        return probs[0, token_to_decode]

class ESMIF1Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class BayesStructModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_model = XLNetWrapper()
        self.seq_struct_model = ProteinMPNNWrapper()

    def forward(self, seq, struct, decode_order, token_to_decode):
        # Ignore the last entry, corresponding to 'X'
        p_seq = self.seq_model(seq=seq, decode_order=decode_order, token_to_decode=token_to_decode)[:-1]
        p_seq_struct = self.seq_struct_model(seq=seq, struct=struct, decode_order=decode_order, token_to_decode=token_to_decode)[:-1]
        p_struct_seq = (p_seq_struct / p_seq)
        # Normalize probabilities
        p_struct_seq = p_struct_seq / p_struct_seq.sum()
        return p_struct_seq

class TrRosettaWrapper():
    def __init__(self, seq_file='tmp.txt', msa_file='msa_tmp.txt', database='/data/uniref30'):
        self.trrosetta = trRosettaNetwork(
            filters = 64,
            kernel = 3,
            num_layers = 61
        ).cuda()
        self.seq_file = seq_file
        self.msa_file = msa_file
        self.database = database

    def __call__(self, seq):
        """Pass a sequence through the trRosetta model and return the distogram
        Args:
            seq (str): a space-separated string representing the amino acid sequence
        Returns:
            distance ((L x L x 37) torch.Tensor): a distogram representing distance bin probabilities
        """
        # Make a fasta file with the sequence
        with open(self.seq_file, 'w') as f:
            f.write('>\n' + ''.join(seq.split()))
        # Get an MSA for the sequence
        os.system(f'hhblits -i {self.seq_file} -oa3m {self.msa_file} -d {self.database}')
        x = preprocess(self.msa_file)
        theta, phi, distance, omega = model(x).squeeze()
        return distance

def test_xlnet_wrapper():
    import random

    seq, struct = get_protein()
    seq = seq[:10]
    given_characters_up_to = 5
    seq = ''.join([char if i < given_characters_up_to else '-' for i, char in enumerate(seq)])

    xlnet = XLNetWrapper()

    decoding_order = np.arange(len(seq)).tolist()
    random.shuffle(decoding_order)
    out_probs = xlnet(seq, decoding_order=decoding_order, token_to_decode=given_characters_up_to)

def test_protein_mpnn_wrapper():
    seq, struct = get_protein()
    given_characters_up_to = 20
    seq = ''.join([char if i < given_characters_up_to else '-' for i, char in enumerate(seq)])

    mpnn = ProteinMPNNWrapper()
    decoding_order = np.arange(len(seq)).tolist()
    out_probs = mpnn(seq=seq, struct=struct, decoding_order=decoding_order, token_to_decode=given_characters_up_to)

model_dict = {'xlnet':XLNetWrapper, 'protein_mpnn':ProteinMPNNWrapper, 'bayes_struct':BayesStructModel, 'trRosetta':TrRosettaWrapper}
