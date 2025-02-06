import numpy as np
import torch
from pathlib import Path
import subprocess
from Bio.PDB import *
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sequence_to_pdb(sequence, coords, output_path):
    """Convert sequence and coordinates to PDB format with full backbone atoms.
    
    Args:
        sequence: String of amino acid one-letter codes
        coords: Array of shape (L, 4, 3) for L residues, 4 atoms (N, CA, C, O), xyz coordinates
        output_path: Path to save PDB file
    """
    logger.info(f"Converting sequence to PDB: {output_path}")
    logger.info(f"Sequence length: {len(sequence)}, Coords shape: {coords.shape}")
    
    # Define backbone atoms
    backbone_atoms = ['N', 'CA', 'C', 'O']
    element_symbols = ['N', 'C', 'C', 'O']
    
    with open(output_path, 'w') as f:
        atom_num = 1
        res_num = 1
        
        for i, (res, pos) in enumerate(zip(sequence, coords)):
            # Skip gaps and missing coordinates
            if res == '-' or (isinstance(pos[0, 0], float) and np.isnan(pos[0, 0])):
                logger.debug(f"Skipping position {i}: res={res}, pos={pos}")
                continue
                
            # Write all backbone atoms for each residue
            for atom_idx, (atom_name, element) in enumerate(zip(backbone_atoms, element_symbols)):
                x, y, z = pos[atom_idx]
                f.write(f"ATOM  {atom_num:5d}  {atom_name:<3} {res:3} A{res_num:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}  \n")
                atom_num += 1
            res_num += 1
            
        f.write("TER\n")
        f.write("END\n")

def get_scaffold_positions(sequence, motif_mask):
    """Get positions that are part of the scaffold (not motif and not gaps)."""
    positions = [i for i, (res, mask) in enumerate(zip(sequence, motif_mask)) 
                if res != '-' and not mask]
    logger.info(f"Found {len(positions)} scaffold positions")
    return positions

def get_residue_mapping(ref_struct, mobile_struct, scaffold_positions):
    """Create mapping between residue positions in two structures."""
    ref_res_map = {}
    mobile_res_map = {}
    
    # Map residue numbers to scaffold positions
    for model in ref_struct:
        for chain in model:
            for res in chain:
                if res.id[1] in scaffold_positions and 'CA' in res:
                    ref_res_map[res.id[1]] = res['CA']
    
    for model in mobile_struct:
        for chain in model:
            for res in chain:
                if res.id[1] in scaffold_positions and 'CA' in res:
                    mobile_res_map[res.id[1]] = res['CA']
    
    # Find common positions
    common_positions = sorted(set(ref_res_map.keys()) & set(mobile_res_map.keys()))
    
    logger.info(f"Reference structure has {len(ref_res_map)} scaffold positions")
    logger.info(f"Mobile structure has {len(mobile_res_map)} scaffold positions")
    logger.info(f"Common positions: {len(common_positions)}")
    
    return [(ref_res_map[pos], mobile_res_map[pos]) for pos in common_positions]

def align_structures(ref_path, mobile_path, scaffold_positions, output_path):
    """Align structures based on scaffold positions."""
    ref_path = os.path.abspath(ref_path)
    mobile_path = os.path.abspath(mobile_path)
    output_path = os.path.abspath(output_path)

    logger.info(f"Aligning structures:")
    logger.info(f"Reference: {ref_path}")
    logger.info(f"Mobile: {mobile_path}")
    
    # Suppress PDB parser warnings
    import warnings
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    warnings.simplefilter('ignore', PDBConstructionWarning)
    
    pdb_parser = PDBParser(QUIET=True)
    ref_struct = pdb_parser.get_structure('ref', ref_path)
    mobile_struct = pdb_parser.get_structure('mobile', mobile_path)
    
    # Get matched pairs of atoms
    atom_pairs = get_residue_mapping(ref_struct, mobile_struct, scaffold_positions)
    
    if not atom_pairs:
        raise ValueError("No matching scaffold positions found between structures")
    
    ref_atoms = [pair[0] for pair in atom_pairs]
    mobile_atoms = [pair[1] for pair in atom_pairs]
    
    logger.info(f"Aligning using {len(ref_atoms)} atom pairs")
    
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, mobile_atoms)
    super_imposer.apply(mobile_struct)
    
    logger.info(f"RMSD after alignment: {super_imposer.rms}")
    
    io = PDBIO()
    io.set_structure(mobile_struct)
    io.save(output_path)

def create_chimerax_script(output_dir, study_name, visualization_type, structures):
    """Create ChimeraX script for visualization."""
    script_path = os.path.join(output_dir, f"{study_name}_visualization.cxc")
    logger.info(f"Creating ChimeraX script: {script_path}")
    
    with open(script_path, 'w') as f:
        f.write("close session\n")
        

        # Load structures with absolute paths
        for name, path in structures.items():
            abs_path = os.path.abspath(path)
            f.write(f"open {abs_path}\n")
        
        if visualization_type == "single_pred":
            # Color scheme for single prediction comparison
            f.write("""
color #1 gray style ribbon
color #2 gray style ribbon
color #3 gray style ribbon
select @CA & #1 & :/motif=true
color sel blue
select @CA & #2 & :/motif=true
color sel green
select @CA & #3 & :/motif=true
color sel red
select clear
""")

        elif visualization_type == "double_pred":
            f.write("""
color #1 gray style ribbon
color #2 gray style ribbon
select @CA & #1 & :/motif=true
color sel green
select @CA & #2 & :/motif=true
color sel light green
select @CA & #3 & :/motif=true
color sel red
select @CA & #4 & :/motif=true
color sel light red
""")
        session_path = os.path.join(output_dir, f"{study_name}.cxs")
        f.write(f"""
view
save {session_path}
""")
    return script_path

def process_study(results, study_info, output_dir):
    """Process a single study and create visualization."""
    logger.info(f"\nProcessing study: {study_info['name']}")
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find relevant result
    result = next(r for r in results 
                 if r["model_name"] == study_info["model"] and 
                 r["domain_name_pro"] == study_info["domain_pro"] and
                 r["domain_name_anti"] == study_info["domain_anti"])
    
    # Convert coordinates to PDB
    pro_pdb = os.path.join(output_dir, "domain_pro.pdb")
    anti_pdb = os.path.join(output_dir, "domain_anti.pdb")
    pred_pdb = os.path.join(output_dir, "pred.pdb")
    
    # Load coordinates from .pt files
    pro_coords = torch.load(result["coords_path_pro"])
    anti_coords = torch.load(result["coords_path_anti"])
    pred_coords = torch.load(result["pred_coords_path"])
    
    logger.info("Coordinate shapes:")
    logger.info(f"Pro: {pro_coords.shape}")
    logger.info(f"Anti: {anti_coords.shape}")
    logger.info(f"Pred: {pred_coords.shape}")
    
    # Convert to PDB
    sequence_to_pdb(result["sequence_pro"], pro_coords, pro_pdb)
    sequence_to_pdb(result["sequence_anti"], anti_coords, anti_pdb)
    sequence_to_pdb(result["pred_sequence"], pred_coords, pred_pdb)
    
    # Get scaffold positions
    scaffold_positions = get_scaffold_positions(result["sequence_pro"], result["motif_mask"])
    
    # Align structures
    aligned_anti = os.path.join(output_dir, "aligned_anti.pdb")
    aligned_pred = os.path.join(output_dir, "aligned_pred.pdb")
    
    align_structures(pro_pdb, anti_pdb, scaffold_positions, aligned_anti)
    align_structures(pro_pdb, pred_pdb, scaffold_positions, aligned_pred)
    
    # Create ChimeraX script
    structures = {
        "pro": pro_pdb,
        "anti": aligned_anti,
        "pred": aligned_pred
    }
    
    script_path = create_chimerax_script(output_dir, study_info["name"], "single_pred", structures)

    # # Run ChimeraX with absolute path
    # subprocess.run(["chimerax", script_path])

def process_study_2(results, study_info, output_dir):
    """Process a single study and create visualization."""
    logger.info(f"\nProcessing study: {study_info['name']}")
    
    # Create output directory
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find relevant results
    # Design for conformation 1
    result_1 = next(r for r in results 
                 if r["model_name"] == study_info["model"] and 
                 r["domain_name_pro"] == study_info["domain_pro"] and
                 r["domain_name_anti"] == study_info["domain_anti"])
    
    # Design for conformation 2
    result_2 = next(r for r in results 
                 if r["model_name"] == study_info["model"] and 
                 r["domain_name_anti"] == study_info["domain_pro"] and
                 r["domain_name_pro"] == study_info["domain_anti"])
    
    # Convert coordinates to PDB
    pro_pdb = os.path.join(output_dir, "domain_pro.pdb")
    anti_pdb = os.path.join(output_dir, "domain_anti.pdb")
    pred_pdb_pro = os.path.join(output_dir, "pred_pro.pdb")
    pred_pdb_anti = os.path.join(output_dir, "pred_anti.pdb")
    
    # Load coordinates from .pt files
    pro_coords = torch.load(result_1["coords_path_pro"])
    anti_coords = torch.load(result_1["coords_path_anti"])
    pred_coords_pro = torch.load(result_1["pred_coords_path"])
    pred_coords_anti = torch.load(result_2["pred_coords_path"])
    
    logger.info("Coordinate shapes:")
    logger.info(f"Pro: {pro_coords.shape}")
    logger.info(f"Anti: {anti_coords.shape}")
    logger.info(f"Pred: {pred_coords_pro.shape}")
    logger.info(f"Pred: {pred_coords_anti.shape}")
    
    # Convert to PDB
    sequence_to_pdb(result_1["sequence_pro"], pro_coords, pro_pdb)
    sequence_to_pdb(result_1["sequence_anti"], anti_coords, anti_pdb)
    sequence_to_pdb(result_1["pred_sequence"], pred_coords_pro, pred_pdb_pro)
    sequence_to_pdb(result_2["pred_sequence"], pred_coords_anti, pred_pdb_anti)
    
    # Get scaffold positions
    scaffold_positions = get_scaffold_positions(result_1["sequence_pro"], result_1["motif_mask"])
    
    # Align structures
    aligned_anti = os.path.join(output_dir, "aligned_anti.pdb")
    aligned_pred_pro = os.path.join(output_dir, "aligned_pred_pro.pdb")
    aligned_pred_anti = os.path.join(output_dir, "aligned_pred_anti.pdb")
    
    align_structures(pro_pdb, anti_pdb, scaffold_positions, aligned_anti)
    align_structures(pro_pdb, pred_pdb_pro, scaffold_positions, aligned_pred_pro)
    align_structures(pro_pdb, pred_pdb_anti, scaffold_positions, aligned_pred_anti)
    
    # Create ChimeraX script
    structures = {
        "pro": pro_pdb,
        "anti": aligned_anti,
        "pred_pro": aligned_pred_pro,
        "pred_anti": aligned_pred_anti
    }
    
    script_path = create_chimerax_script(output_dir, study_info["name"], "single_pred", structures)

    # # Run ChimeraX with absolute path
    # subprocess.run(["chimerax", script_path])

def process_algorithm_comparison(results, study_info, output_dir):
    """Process algorithm comparison studies."""
    logger.info(f"\nProcessing algorithm comparison: {study_info['name']}")
    os.makedirs(output_dir, exist_ok=True)
    
    for model in ["cs_design", "protein_mpnn"]:
        result = next(r for r in results 
                     if r["model_name"] == model and 
                     r["domain_name_pro"] == study_info["domain_pro"] and
                     r["domain_name_anti"] == study_info[f"{model}_domain_anti"])
        
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        process_study(results, {
            "name": f"{study_info['name']}_{model}",
            "model": model,
            "domain_pro": study_info["domain_pro"],
            "domain_anti": study_info[f"{model}_domain_anti"]
        }, model_dir)

# Example usage remains the same...

import json
import argparse
# Example usagedd
def main(args):

    with open(os.path.join(args.output_dir, "predictions.txt"), "r") as f:
        results = json.load(f)

    # Best Individual Design (Smallest RMSD_pro)
    # Define studies
    studies = [
        {
            "name": "best_individual_cs",
            "model": "cs_design",
            "domain_pro": "2x7rB00",
            "domain_anti": "3cp1A00"
        },
        {
            "name": "best_individual_protein_mpnn",
            "model": "protein_mpnn",
            "domain_pro": "2x7rB00",
            "domain_anti": "3cp1A00"
        },
    ]

    studies_2 = [
        {
            "name": "best_overall_cs",
            "model": "cs_design",
            "domain_pro": "1akjB00",
            "domain_anti": "5csbA00"
        },
        {
            "name": "best_overall_mpnn",
            "model": "protein_mpnn",
            "domain_pro": "2odhA01",
            "domain_anti": "3imbB01"
        },
    ]
    
    algorithm_comparisons = [
        {
            "name": "largest_algorithm_diff_cs_design",
            "domain_pro": "2axzA02",
            "cs_design_domain_anti": "2awiD02",
            "protein_mpnn_domain_anti": "2awiD02"
        },
        {
            "name": "largest_algorithm_diff_protein_mpnn",
            "domain_pro": "4jkaB01",
            "cs_design_domain_anti": "4jkfA01",
            "protein_mpnn_domain_anti": "4jkfA01"
        },
    ]
    
    base_output_dir = "visualization_output"
    
    # Process individual studies
    # for study in studies:
    #     study_dir = os.path.join(base_output_dir, study["name"])
    #     process_study(results, study, study_dir)

    for study in studies_2:
        study_dir = os.path.join(base_output_dir, study["name"])
        process_study_2(results, study, study_dir)

    
    # Process algorithm comparisons
    for comparison in algorithm_comparisons:
        comparison_dir = os.path.join(base_output_dir, comparison["name"])
        process_algorithm_comparison(results, comparison, comparison_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    args = parser.parse_args()
    main(args)