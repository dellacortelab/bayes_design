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
from pathlib import Path
import subprocess
from transformers import XLNetTokenizer, XLNetLMHeadModel
from protein_mpnn_utils import ProteinMPNN
from tr_rosetta_pytorch import trRosettaNetwork
from tr_rosetta_pytorch.utils import preprocess
from tr_rosetta_pytorch.cli import DEFAULT_MODEL_PATH

from .utils import AMINO_ACID_ORDER, get_protein

class XLNetWrapper():
    def __init__(self, model_name='Rostlab/prot_xlnet', device=None):
        
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.model = XLNetLMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)

        xlnet_vocab_dict = self.tokenizer.get_vocab()
        xlnet_vocab_dict['▁X'] = xlnet_vocab_dict['X']
        self.canonical_idx_to_xlnet_idx = torch.tensor([xlnet_vocab_dict['▁' + aa] for aa in AMINO_ACID_ORDER])

    def __call__(self, seq, decode_order, token_to_decode, struct=None, mask_type='bidirectional_autoregressive'):
        """Accept an amino acid sequence, return class probabilities for the next token
        Args:
            seq (len 1 list of len L_seq str): a string representation of an amino 
                acid sequence with unknown residues indicated with a dash (-)
            decode_order (len L list): list of the order of indices to decode.
                This determines the values in the permutation mask. Each index 
                attends to all indices that occur previously in the decoding_order.
            token_to_decode (int or len N tensor): index in the range [0, L-1] indicating which 
                token to predict.
        Returns:
            probs ((20) torch.Tensor): a vector of probabilities for the next
                token
        """
        # When using bidirection autoregressive, we should get the same probabilities whether providing the whole sequence
        # or the masked sequence, because the permutation mask protects us.
        # When using unidirectional autoregressive, we should get the same probabilities as well.
        # When using bidirectional mlm, we should get different probabilities as we take advantage of the provided
        # future context in one case and not the other
        
        # Replace rare amino acids with "X"
        seq = [re.sub(r"[UZOB]", "X", s) for s in seq]
        # Huggingface XLNet expects a space-separated sequence
        seq = [" ".join(s) for s in seq]
        seq = [re.sub(r"-", "<mask>", s) for s in seq]
        input_ids = self.tokenizer(seq, return_tensors='pt', add_special_tokens=False)['input_ids']
        seq_len = input_ids.shape[-1]
        # Mask '-' tokens
        
        if not hasattr(token_to_decode, '__len__'):
            token_to_decode = [token_to_decode]

        n_tokens = len(token_to_decode)

        # perm_mask should be the mask for content stream attention (allow items to see self), because the xlnet 
        # implementation adds an identity matrix to perm_mask to get the query stream attention, where items are
        # masked from seeing self. The target_mapping ensures that we use query stream attention when predicting
        # the target elements https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/models/xlnet/modeling_xlnet.py#L1172
        perm_mask = torch.ones((n_tokens, seq_len, seq_len), dtype=torch.float) # perm_mask[0, j, k] = 1 means that the jth token cannot see the kth token
        if mask_type == 'unidirectional_autoregressive':
            # Allow each token to see tokens preceding it in decode order
            for i, tok in enumerate(token_to_decode):
                token_to_decode_idx = decode_order.index(tok)
                # This is + 1 because we decide what the decoded item sees
                for j, idx in enumerate(decode_order[:token_to_decode_idx + 1]):
                    # This should be j+1, because we allow tokens to see themselves
                    perm_mask[i, idx, decode_order[:j+1]] = 0.0
            # decode_order: [2, 1, 0]
            # i = 0 -> decode pos 2
            #   j = 0, idx = 2  
            #       perm_mask[0, 2, [2]] = 0
            #           [1, 1, 1]
            #           [1, 1, 1]
            #           [1, 1, 0]
            # i = 1 -> decode pos 1
            #   j = 0, idx = 2
            #       perm_mask[1, 2, [2]] = 0.
            #   j = 1, idx = 1
            #       perm_mask[1, 1, [2, 1]] = 0.
            #           [1, 1, 1]
            #           [1, 0, 0]
            #           [1, 1, 0]
            # i = 2 -> decode pos 0
            #   j = 0, idx = 2
            #       perm_mask[2, 2, [2]] = 0.
            #   j = 1, idx = 1
            #       perm_mask[2, 1, [2, 1]] = 0.
            #           [1, 1, 1]
            #           [1, 0, 0]
            #           [1, 1, 0]
            #   j = 2, idx = 0
            #       perm_mask[2, 0, [2, 1, 0]] = 0.
            #           [0, 0, 0]
            #           [1, 0, 0]
            #           [1, 1, 0]
        elif mask_type == 'bidirectional_autoregressive':
            for i, tok in enumerate(token_to_decode):
                token_to_decode_idx = decode_order.index(tok)
                for prev_tok_1 in decode_order[:token_to_decode_idx]:
                    for prev_tok_2 in decode_order[:token_to_decode_idx]:
                        # Allow all tokens to see tokens preceding token_to_decode
                        perm_mask[i, prev_tok_1, prev_tok_2] = 0.0
                perm_mask[i, tok, decode_order[:token_to_decode_idx+1]] = 0.0
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 0]
            #
            #       [1, 1, 1, 1]
            #       [1, 1, 1, 1]
            #       [1, 1, 0, 0]
            #       [1, 1, 1, 0]
            #
            #       [1, 1, 1, 1]
            #       [1, 0, 0, 0]
            #       [1, 1, 0, 0]
            #       [1, 1, 0, 0]
            #
            #       [0, 0, 0, 0]
            #       [1, 0, 0, 0]
            #       [1, 0, 0, 0]
            #       [1, 0, 0, 0]
        elif mask_type == 'bidirectional_mlm':
            for i, tok in enumerate(token_to_decode):
                perm_mask[i, :, np.arange(seq_len) != tok] = 0.0 # Allow full bidirectional context except for the decoded token (masked-language-model-style. this is not autoregressive)
            # decode_order: [2, 1, 0]
            # i = 0 -> decode pos 2
            # perm_mask[0, :, [1, 0]] = 0.0
            # i = 1 -> decode pos 1
            # perm_mask[1, :, [2, 0]] = 0.0
            # i = 2 -> decode pos 0
            # perm_mask[2, :, [2, 1]] = 0.0

        target_mapping = torch.zeros(
            (n_tokens, 1, seq_len), dtype=torch.float
        )  # Shape [batch_size=n_tokens, num_tokens_to_predict=1, seq_length=n_tokens]
        for i, tok in enumerate(token_to_decode):
            target_mapping[i, 0, tok] = 1.0 # Predict the ith token

        with torch.inference_mode():
            out = self.model(input_ids.to(self.device), perm_mask=perm_mask.to(self.device), target_mapping=target_mapping.to(self.device))
            # logits has shape [batch_size, 1, config.vocab_size]
            index_corrected_logits = out.logits[:, 0, self.canonical_idx_to_xlnet_idx]
            # Ignore the last entry, corresponding to 'X'
            index_corrected_logits = index_corrected_logits[:, :-1]
            probs = torch.nn.functional.softmax(index_corrected_logits, dim=-1)
        
        return probs


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
    def __init__(self, device=None):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        #v_48_010=version with 48 edges 0.10A noise
        model_name = "v_48_030"
        backbone_noise=0.00               # Standard deviation of Gaussian noise to add to backbone atoms
        path_to_model_weights='/root/ProteinMPNN/vanilla_proteinmpnn/vanilla_model_weights'          
        hidden_dim = 128
        num_layers = 3 
        model_folder_path = path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
        checkpoint_path = model_folder_path + f'{model_name}.pt'
        checkpoint = torch.load(checkpoint_path, map_location=self.device) 
        print('Number of edges:', checkpoint['num_edges'])
        noise_level_print = checkpoint['noise_level']
        print(f'Training noise level: {noise_level_print}')
        self.model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Model loaded")
    
    def forward(self, seq, struct, decode_order, token_to_decode):
        """Accept an amino acid sequence and protein structure 
        coordinates, return class probabilities for the next token
        Args:
            seq (len N list of len L_seq str): a string representation of an amino 
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
            probs ((N x 20) torch.Tensor): a vector of probabilities for the next
                token
        """
        N = len(seq)
        L = len(seq[0])
        assert L == struct.shape[0], "Sequence length must match the number of residues in the provided structure"
        # Convert amino acid character to index
        seq = [re.sub(r"-", "X", s) for s in seq]
        seq = torch.tensor([[AMINO_ACID_ORDER.index(aa) for aa in s] for s in seq]).to(self.device)
        struct = struct.expand(N, *struct.shape).to(self.device)
        with torch.no_grad():
            # Default values
            mask = torch.ones(N, L).float().to(self.device)
            chain_M = torch.ones(N, L).float().to(self.device)
            chain_encoding_all = torch.ones(N, L).float().to(self.device)
            residue_idx = torch.arange(L).expand(N, L).to(self.device)

            # randn determines the decoding order. The indices in the randn vector with the lowest values will be decoded first
            randn = torch.argsort(torch.tensor(decode_order)).to(self.device)
            log_probs = self.model(X=struct, S=seq, mask=mask, chain_M=chain_M, residue_idx=residue_idx, chain_encoding_all=chain_encoding_all, randn=randn, use_input_decoding_order=True, decoding_order=torch.tensor(decode_order).expand(N, L).to(self.device))
            probs = torch.exp(log_probs)
            # N x L x 20
            # Ignore the last entry, corresponding to 'X', and normalize
            probs = probs[:, :, :-1] / probs[:, :, :-1].sum(dim=-1).unsqueeze(-1)

        if hasattr(token_to_decode, '__len__'):
            return probs[range(len(token_to_decode)), token_to_decode]
            # N x 20
        else:
            return probs[:, token_to_decode]
            # N x 20

        #     # Required for sampling
        #     chain_M_pos = torch.ones(N, L).float().to(self.device)
        #     bias_AAs_np = np.zeros(len(AMINO_ACID_ORDER))
        #     bias_by_res = torch.zeros([N, L, len(AMINO_ACID_ORDER)]).to(self.device)
        #     # Predict tokens except 'X'
        #     omit_AAs_np = np.array([AA in ['X'] for AA in AMINO_ACID_ORDER]).astype(np.float32)
        #     out = self.model.sample(X=struct, randn=randn, S_true=seq, chain_mask=chain_M, chain_encoding_all=chain_encoding_all, residue_idx=residue_idx, mask=mask, chain_M_pos=chain_M_pos, bias_AAs_np=bias_AAs_np, omit_AAs_np=omit_AAs_np, bias_by_res=bias_by_res)
        #     probs = out['probs']
        # # Return the probabilities for th 0th chain and the 0th residue
        # return probs[:, token_to_decode]

class ESMIF1Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

class BayesStructModel(nn.Module):
    def __init__(self, device=None, bayes_balance_factor=0.002, **kwargs):
        super().__init__()
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.seq_model = XLNetWrapper(**kwargs)
        self.seq_struct_model = ProteinMPNNWrapper(**kwargs)

        self.bayes_balance_factor = bayes_balance_factor

    def forward(self, seq, struct, decode_order, token_to_decode, mask_type='bidirectional_autoregressive'):
        p_seq = self.seq_model(seq=seq, decode_order=decode_order, token_to_decode=token_to_decode, mask_type=mask_type).clone()
        p_seq_struct = self.seq_struct_model(seq=seq, struct=struct, decode_order=decode_order, token_to_decode=token_to_decode).clone()

        # unbalanced_logits = (p_seq_struct / p_seq)
        # p_struct_seq_1 = unbalanced_logits / unbalanced_logits.sum(dim=-1).unsqueeze(-1)
        # torch.set_printoptions(sci_mode=False)
        # print("seq struct")
        # print(p_seq_struct[0])
        # print('seq')
        # print(p_seq[0])
        # print('unbalanced logits')
        # print((unbalanced_logits)[0])
        # print('unbalanced normalized probs')
        # print(p_struct_seq_1[0])

        # Add a "balance factor" so that we don't end up with large probability ratios at the tails of the distributions
        p_seq += self.bayes_balance_factor
        p_seq_struct += self.bayes_balance_factor
        balanced_logits = (p_seq_struct / p_seq)
        
        # Normalize probabilities
        p_struct_seq = balanced_logits / balanced_logits.sum(dim=-1).unsqueeze(-1)
        # print("seq struct")
        # print(p_seq_struct[0])
        # print('seq')
        # print(p_seq[0])
        # print('balanced_logits')
        # print((balanced_logits)[0])
        # print('balanced normalized probs')
        # print(p_struct_seq[0])
        # import pdb; pdb.set_trace()

        return p_struct_seq

class TrRosettaWrapper():
    def __init__(self, data_location='/data/msa', database='/data/uniref30/UniRef30_2022_02', device=None):

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')
        
        model_files = [*Path(DEFAULT_MODEL_PATH).glob('*.pt')]
        self.models = []
        for model_file in model_files[:1]:
            trrosetta = trRosettaNetwork(
                filters = 64,
                kernel = 3,
                num_layers = 61
            ).to(self.device)
            trrosetta.load_state_dict(torch.load(model_file, map_location=self.device))
            trrosetta.eval()
            self.models.append(trrosetta)

        self.data_location = data_location
        os.makedirs(self.data_location, exist_ok=True)

        self.database = database

    def __call__(self, seq, seq_id):
        """Pass a sequence through the trRosetta model and return the distogram
        Args:
            seq (str): a space-separated string representing the amino acid sequence
        Returns:
            distance ((L x L x 37) torch.Tensor): a distogram representing distance bin probabilities
        """
        seq_path = os.path.join(self.data_location, f'{seq_id}.txt')
        seq_msa_path = os.path.join(self.data_location, f'{seq_id}_msa.txt')
        # Make a fasta file with the sequence
        with open(seq_path, 'w') as f:
            f.write('>\n' + ''.join(seq.split()))
        # Get an MSA for the sequence
        if not os.path.exists(seq_msa_path):
            out = subprocess.run(['/root/hh-suite/bin/hhblits', '-i', f'{seq_path}', '-oa3m', f'{seq_msa_path}', '-d', f'{self.database}'])
        x = preprocess(seq_msa_path).to(self.device)
        outputs = []
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                outputs.append(output)
            averaged_outputs = [torch.stack(model_output).mean(dim=0).cpu().numpy().squeeze(0).transpose(1,2,0) for model_output in zip(*outputs)]
            # prob_theta, prob_phi, prob_distance, prob_omega
            output_dict = dict(zip(['theta', 'phi', 'dist', 'omega'], averaged_outputs))
            distance = output_dict['dist']
            distance = distance.squeeze()

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

model_dict = {'xlnet':XLNetWrapper, 'protein_mpnn':ProteinMPNNWrapper, 'bayes_design':BayesStructModel, 'trRosetta':TrRosettaWrapper}
