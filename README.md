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
python3 design.py --model_name bayes_design --protein_id 6MRR --decode_order n_to_c --decode_algorithm beam --n_beams 4 --fixed_positions 67 68 
```

## To try other experiments:
### Evaluate the log probability of a sequence under a given sequence design
```
python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG MLPEGWKLIKDPKTGQDMCFNEITKEITAQRPVG KLPEGWVEVTDPKTGKKLYYNTKTKEITEEKPVG
```
### Evaluate the trRosetta log probability of the true structure for a designed sequence
```
python3 experiment.py compare_struct_probs --protein_id 1PIN --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG MLPEGWKLIKDPKTGQDMCFNEITKEITAQRPVG KLPEGWVEVTDPKTGKKLYYNTKTKEITEEKPVG
```
