import argparse

def main(args):
    import json
    import torch 
    from .dataset import worker_init_fn, get_pdbs, loader_pdb, build_training_clusters, PDB_dataset 


    data_path = args.path_for_training_data
    params = {
        "LIST"    : f"{data_path}/list.csv", 
        "VAL"     : f"{data_path}/valid_clusters.txt",
        "TEST"    : f"{data_path}/test_clusters.txt",
        "DIR"     : f"{data_path}",
        "DATCUT"  : "2030-Jan-01",
        "RESCUT"  : args.rescut, #resolution cutoff for PDBs
        "HOMO"    : 0.70 #min seq.id. to detect homo chains
    }


    LOAD_PARAM = {'batch_size': 1,
                  'shuffle': True,
                  'pin_memory':False,
                  'num_workers': 0}

   
    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, test = build_training_clusters(params, args.debug)
     
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(train_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(valid_set, worker_init_fn=worker_init_fn, **LOAD_PARAM)

    # Save the validation dataset
    pdb_dict_valid=get_pdbs(valid_loader,1,args.max_protein_length,args.num_examples_per_epoch)
    with open("pdb_dict_valid.jsonl", "w") as jsonl_file:
        # Write each dictionary as a separate JSON object on a new line
        for pdb_dict in pdb_dict_valid:
            json.dump(pdb_dict, jsonl_file)
            jsonl_file.write('\n')  # Add a newline to separate JSON objects


    # Save the train dataset
    pdb_dict_train=get_pdbs(train_loader,1,args.max_protein_length,args.num_examples_per_epoch)
    with open("pdb_dict_train.jsonl", "w") as jsonl_file:
        # Write each dictionary as a separate JSON object on a new line
        for pdb_dict in pdb_dict_train:
            json.dump(pdb_dict, jsonl_file)
            jsonl_file.write('\n')  # Add a newline to separate JSON objects
   


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02", help="path for loading training data")
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000, help="number of training example to load for one epoch")
    argparser.add_argument("--max_protein_length", type=int, default=10000, help="maximum length of the protein complext")
    argparser.add_argument("--rescut", type=float, default=3.5, help="PDB resolution cutoff")
    argparser.add_argument("--debug", type=bool, default=False, help="minimal data loading for debugging")
    args = argparser.parse_args()    
    main(args)