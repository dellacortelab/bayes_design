# enzyme-design
# Use cases
## Running with Docker
- Clone repository
```
git clone git@github.com:dellacortelab/enzyme-design.git 
cd docking-to-publish
```
- To build container
```
docker build -t jacobs-enzyme-design -f dependencies/Dockerfile dependencies
```
- To run container
```
docker run -dit --gpus all --name jacobs-enzyme-dev -v $(pwd):/code -v /mnt/pccfs2/not_backed_up/jacobastern/enzyme-design:/data jacobs-enzyme-design
docker exec -it jacobs-enzyme-dev /bin/bash
```
- Training a model
```
python3 ensemble_dock_train.py --model_save_path ./results/models/dude-lit-pcba.pt --train_dataset dude-lit-pcba --val_dataset dude-lit-pcba --device 0
```
- Fine-tuning a model
```
python3 ensemble_dock_train.py --model_load_path ./results/models/dude.pt --model_save_path ./results/models/dude-base-lit-pcba-finetune.pt --train_dataset lit-pcba --val_dataset dude lit-pcba --device 1
```
- Testing a model and placing the results in the results .pkl
```
python3 ensemble_dock_test.py --model_load_path ./results/models/dude-lit-pcba.pt --dataset dude-lit-pcba --device 0
```
- Evaluating a model to make figure 1
```
python3 ensemble_dock_figures.py --model_names dude-lit-pcba vina_top_score --dataset dude-mini --test_fig fig_1
```
