# Explanation

There are two possibilites for a training script. 

One is to repurpose the XLNet training script. This has the benefit of ensuring that masking is done properly.

The other is to repurpose the ProteinMPNN training script. This has the benefit of ensuring that the dataset splits are handled properly.

We are currently going with #2.

python3 train_xlnet.py



# Notes
Effect of permutation on labels: permutation mask is used to determine which tokens a given token can attend to.
Are labels permuted? No, labels are not permuted. The permutation mask is used to determine which tokens a given
token can attend to. The labels are used to compute the loss, and are not permuted.

For normal XLNet: 
At train time, all unmasked tokens can see each other. Masked tokens can see all unmasked tokens, and can see
masked tokens to the left of them in the decode order. ("bidirectional autoregressive")

For normal ProteinMPNN:
At train time, all tokens are masked and predicted. Masked tokens can see all tokens to the left of them. ("unidirectional autoregressive")

Differences: 
We might need to make sure that mask and chain_M from ProteinMPNN become part of the XLNet mask

Todo: figure out what 1/0 mean for non_func_mask and make sure that the valid mask is used properly
Todo: make sure all non-aa tokens are mapped correctly, including <pad>, etc.
