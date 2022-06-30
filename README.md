# bayes_design
## Steps to run with Docker
- Clone repository
```
git clone git@github.com:dellacortelab/enzyme-design.git 
cd docking-to-publish
```
- To build container
```
docker build -t jacobs_bayes_design -f dependencies/Dockerfile dependencies
```
- To run container
```
docker run -dit --gpus all --name jacobs_bayes_dev --rm -v $(pwd):/code -v /mnt/pccfs2/not_backed_up/jacobastern/bayes_design:/data jacobs_bayes_design
docker exec -it jacobs_bayes_dev /bin/bash
```
- Redesign a protein backbone
```
python3 design.py --model_name bayes_struct --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 4 --fixed_positions 67 68 
```
