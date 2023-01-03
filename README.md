# BayesDesign
<img src="https://github.com/dellacortelab/bayes_design/blob/master/data/figs/bayes_design.png?raw=true" alt="drawing" width="700"/>

BayesDesign is an algorithm for designing proteins with high stability and conformational specificity. See [preprint here](https://www.biorxiv.org/content/10.1101/2022.12.28.521825v1?rss=1).

Try out the BayesDesign model here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/bayes_design/blob/main/src/bayes_design.ipynb)

Dependencies: `./dependencies/requirements.txt`.

## One-line sequence design
To design a protein sequence to fit a protein backbone:
```
python3 ./code/design.py --model_name bayes_design --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 67 68
```

## Detailed steps to run with Docker
- Clone repository
```
git clone https://github.com/dellacortelab/bayes_design.git
```
- Build container
```
docker build -t bayes_design -f ./bayes_design/dependencies/Dockerfile ./bayes_design/dependencies
```
- Run container
```
docker run -dit --gpus all --name bayes_dev --rm -v $(pwd)/bayes_design:/code -v $(pwd)/bayes_design/data:/data bayes_design
docker exec -it bayes_dev /bin/bash
```
- Redesign a protein backbone
```
python3 ./code/design.py --model_name bayes_design --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 128 --fixed_positions 67 68
```
## Citation
```
@article {Stern2022.12.28.521825,
	author = {Stern, Jacob A. and Free, Tyler J. and Stern, Kimberlee L. and Gardiner, Spencer and Dalley, Nicholas A. and Bundy, Bradley C. and Price, Joshua L. and Wingate, David and Corte, Dennis Della},
	title = {A probabilistic view of protein stability, conformational specificity, and design},
	elocation-id = {2022.12.28.521825},
	year = {2022},
	doi = {10.1101/2022.12.28.521825},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/12/30/2022.12.28.521825},
	eprint = {https://www.biorxiv.org/content/early/2022/12/30/2022.12.28.521825.full.pdf},
	journal = {bioRxiv}
}
```
