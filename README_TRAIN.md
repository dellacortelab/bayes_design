# Explanation

There are two possibilites for a training script. 

One is to repurpose the XLNet training script. This has the benefit of ensuring that masking is done properly.

The other is to repurpose the ProteinMPNN training script. This has the benefit of ensuring that the dataset splits are handled properly.

We are currently going with #2.

python3 train_xlnet.py