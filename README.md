# BayesDesign
<img src="https://github.com/dellacortelab/bayes_design/blob/master/data/figs/bayes_design.png?raw=true" alt="drawing" width="700"/>

BayesDesign is an algorithm for designing proteins with high stability and conformational specificity. See [preprint here](https://www.biorxiv.org/content/10.1101/2022.12.28.521825v1?rss=1).

Try out the BayesDesign model here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/bayes_design/blob/master/examples/BayesDesign.ipynb)

## Requirements
- micromamba

## Installation
```
# Micromamba install command (run and follow prompts)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# Clone BayesDesign repo
git clone https://github.com/dellacortelab/bayes_design.git
# Create environment
cd bayes_design
micromamba env create -f bayes_design.yaml
```

## One-line sequence design
To design a protein sequence to fit a protein backbone:
```
micromamba activate bayes_design
python3 design.py --model_name bayes_design --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 67 68
```

## Other examples
- Redesign a protein to prefer one conformation (CSDesign)
```
python3 -m bayes_design.design --model_name cs_design --protein_id 4GSB --protein_id_anti 2ERK --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 16 16 31 34 52 52 62 62 65 65 68 69 147 165 183 184
```
- CSDesign with a spatial mask while preserving all residues within an 8-angstrom radius of fixed positions
```
python3 -m bayes_design.design --model_name cs_design --protein_id 4GSB --protein_id_anti 2ERK --decode_order n_to_c --decode_algorithm greedy --fixed_positions 1 168 187 358 --ball_mask
```
python3 -m bayes_design.experiments.scrmsd_experiment