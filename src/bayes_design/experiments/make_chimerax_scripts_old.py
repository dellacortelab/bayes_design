from Bio.PDB import PDBIO, StructureBuilder
import os
import json
import argparse
import numpy as np
import torch

def load_coords(file_path):
    """Load coordinates from a file using torch."""
    return torch.load(file_path).numpy()  # Convert to NumPy array for easier handling

def create_pdb(sequence, coords, filename):
    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("pdb")
    sb.init_model(0)
    sb.init_chain("A")
    
    for i, (residue, coord) in enumerate(zip(sequence, coords)):
        # Skip residues with missing coordinates
        if np.isnan(coord).any() or len(coord) != 3:
            print(f"Skipping residue {i + 1} due to missing or invalid coordinates.")
            continue
        
        # Initialize residue
        sb.init_seg(' ')
        sb.init_residue('UNK', ' ', i + 1, '')  # Residue number starts at 1
        
        # Add the CA atom
        sb.init_atom(
            name='CA',  # Atom name (must be 1-4 characters)
            coord=coord,  # Coordinates (x, y, z)
            b_factor=1.0,  # B-factor
            occupancy=1.0,  # Occupancy
            altloc=' ',  # Alternate location indicator
            fullname='CA',  # Full atom name
            serial_number=i + 1,  # Atom serial number
            element='C'  # Element symbol
        )
    
    io = PDBIO()
    io.set_structure(sb.get_structure())
    io.save(filename)

def extract_and_convert(results, study):
    for result in results:
        if result['domain_name_pro'] == study['Domain Pro'] and result['domain_name_anti'] == study['Domain Anti']:
            # Load coordinates from file paths
            coords_pro = load_coords(result['coords_path_pro'])
            coords_anti = load_coords(result['coords_path_anti'])
            pred_coords = load_coords(result['pred_coords_path'])
            
            # Ensure coords are in the correct format (N x 3)
            coords_pro = np.array(coords_pro)[:, 1, :]
            coords_anti = np.array(coords_anti)[:, 1, :]
            pred_coords = np.array(pred_coords)[:, 1, :]
            
            # Create PDB files
            create_pdb(result['sequence_pro'], coords_pro, f"{study['Domain Pro']}.pdb")
            create_pdb(result['sequence_anti'], coords_anti, f"{study['Domain Anti']}.pdb")
            create_pdb(result['pred_sequence'], pred_coords, f"pred_{study['Domain Pro']}_{study['Domain Anti']}.pdb")
            break

def generate_chimerax_script(study, study_type):
    script = f"""
open {study['Domain Pro']}.pdb
open {study['Domain Anti']}.pdb
open pred_{study['Domain Pro']}_{study['Domain Anti']}.pdb
"""
    
    if study_type == "Best Individual Design":
        script += f"""
align #{study['Domain Pro']} & ~mask to #{study['Domain Anti']} & ~mask
color scaffold_color #{study['Domain Pro']} & ~mask
color scaffold_color #{study['Domain Anti']} & ~mask
color blue #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
color green #{study['Domain Pro']} & mask
color red #{study['Domain Anti']} & mask
"""
    elif study_type == "Best Overall Design":
        script += f"""
align #{study['Domain Pro']} & ~mask to #{study['Domain Anti']} & ~mask
align #{study['Domain Pro']} & ~mask to #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color scaffold_color #{study['Domain Pro']} & ~mask
color scaffold_color #{study['Domain Anti']} & ~mask
color scaffold_color #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color green #{study['Domain Pro']} & mask
color lightgreen #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
color red #{study['Domain Anti']} & mask
color lightred #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
"""
    elif study_type == "Largest Algorithm Difference":
        script += f"""
align #{study['Domain Pro']} & ~mask to #{study['Domain Anti']} & ~mask
align #{study['Domain Pro']} & ~mask to #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color scaffold_color #{study['Domain Pro']} & ~mask
color scaffold_color #{study['Domain Anti']} & ~mask
color scaffold_color #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color blue #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
color green #{study['Domain Pro']} & mask
color red #{study['Domain Anti']} & mask
"""
    elif study_type == "Largest Algorithm Summed Difference":
        script += f"""
align #{study['Domain Pro']} & ~mask to #{study['Domain Anti']} & ~mask
align #{study['Domain Pro']} & ~mask to #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color scaffold_color #{study['Domain Pro']} & ~mask
color scaffold_color #{study['Domain Anti']} & ~mask
color scaffold_color #pred_{study['Domain Pro']}_{study['Domain Anti']} & ~mask
color green #{study['Domain Pro']} & mask
color lightgreen #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
color red #{study['Domain Anti']} & mask
color lightred #pred_{study['Domain Pro']}_{study['Domain Anti']} & mask
"""
    
    with open(f"{study_type}_{study['Domain Pro']}_{study['Domain Anti']}.cxc", "w") as f:
        f.write(script)

def main(args):
    studies = [
        {
            "type": "Best Individual Design",
            "Domain Pro": "2x7rB00",
            "Domain Anti": "3cp1A00",
            "RMSD Pro": 0.501
        },
        {
            "type": "Best Overall Design",
            "Domain Pro": "1akjB00",
            "Domain Anti": "5csbA00",
            "Total RMSD": 1.274
        },
        {
            "type": "Largest Algorithm Difference",
            "Domain Pro": "2axzA02",
            "CSDesign Domain Anti": "2awiD02",
            "ProteinMPNN Domain Anti": "2awiD02",
            "CSDesign RMSD Difference": 11.141,
            "ProteinMPNN RMSD Difference": -16.128,
            "Absolute Difference between Algorithms": 15.443
        },
        {
            "type": "Largest Algorithm Summed Difference",
            "Domain Pro": "2awiD02",
            "CSDesign Domain Anti": "2axzA02",
            "ProteinMPNN Domain Anti": "2axzA02",
            "CSDesign Summed RMSD": 8.675,
            "ProteinMPNN Summed RMSD": 24.119,
            "Absolute Difference between Algorithms": 15.443
        }
    ]
    
    with open(os.path.join(args.output_dir, "predictions.txt"), "r") as f:
        results = json.load(f)

    for study in studies:
        extract_and_convert(results, study)
        generate_chimerax_script(study, study['type'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    args = parser.parse_args()
    main(args)