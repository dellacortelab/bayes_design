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
    """Convert sequence and coordinates to PDB format with full backbone atoms and SEQRES records.
    SEQRES includes all residues, while ATOM records are only written for residues with coordinates.
    
    Args:
        sequence: String of amino acid one-letter codes
        coords: Array of shape (L, 4, 3) for L residues, 4 atoms (N, CA, C, O), xyz coordinates
        output_path: Path to save PDB file
    """
    # logger.info(f"Converting sequence to PDB: {output_path}")
    # logger.info(f"Sequence length: {len(sequence)}, Coords shape: {coords.shape}")
    
    # Define backbone atoms
    backbone_atoms = ['N', 'CA', 'C', 'O']
    element_symbols = ['N', 'C', 'C', 'O']
    
    # Define amino acid 3-letter codes for SEQRES
    aa_codes = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR', 'X':'UNK'
    }
    
    with open(output_path, 'w') as f:
        # Write SEQRES records for complete sequence (13 residues per line)
        residues_per_line = 13
        num_seqres_lines = (len(sequence) + residues_per_line - 1) // residues_per_line
        
        for i in range(num_seqres_lines):
            start_idx = i * residues_per_line
            end_idx = min(start_idx + residues_per_line, len(sequence))
            residues = sequence[start_idx:end_idx]
            
            # Convert to 3-letter codes and join with spaces
            res_line = ' '.join(aa_codes[res] for res in residues)
            
            # Write SEQRES line with proper formatting
            f.write(f"SEQRES  {i+1:2d} A {len(sequence):4d}  {res_line:<39s}\n")
        
        atom_num = 1
        res_num = 1
        
        # Write ATOM records only for residues with coordinates
        for i, (res, pos) in enumerate(zip(sequence, coords)):
            # Skip residues with missing coordinates
            if torch.isnan(pos.sum()):
                res_num += 1
                continue
                
            # Write all backbone atoms for residue
            for atom_idx, (atom_name, element) in enumerate(zip(backbone_atoms, element_symbols)):
                x, y, z = pos[atom_idx]
                f.write(f"ATOM  {atom_num:5d}  {atom_name:<3} {aa_codes[res]:3} A{res_num:4d}    "
                       f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element}  \n")
                atom_num += 1
            res_num += 1
            
        f.write("TER\n")
        f.write("END\n")

def get_scaffold_positions(motif_mask, coords_1, coords_2):
    """Get positions that are part of the scaffold (not motif and not gaps)."""
    positions = [i+1 for i, (mask, coord_1, coord_2) in enumerate(zip(motif_mask, coords_1, coords_2)) 
                if not torch.isnan(coord_1.sum(-1).sum(-1)) and not torch.isnan(coord_2.sum(-1).sum(-1)) and not mask]
    # logger.info(f"Found {len(positions)} non-nan scaffold positions")
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
    
    # logger.info(f"Reference structure has {len(ref_res_map)} scaffold positions")
    # logger.info(f"Mobile structure has {len(mobile_res_map)} scaffold positions")
    # logger.info(f"Common positions: {len(common_positions)}")
    
    return [(ref_res_map[pos], mobile_res_map[pos]) for pos in common_positions]

def align_structures(ref_path, mobile_path, scaffold_positions, output_path):
    """Align structures based on scaffold positions."""
    ref_path = os.path.abspath(ref_path)
    mobile_path = os.path.abspath(mobile_path)
    output_path = os.path.abspath(output_path)

    # logger.info(f"Aligning structures:")
    # logger.info(f"Reference: {ref_path}")
    # logger.info(f"Mobile: {mobile_path}")
    
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
    
    # logger.info(f"Aligning using {len(ref_atoms)} atom pairs")
    
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, mobile_atoms)
    super_imposer.apply(mobile_struct)
    
    # logger.info(f"RMSD after alignment: {super_imposer.rms}")
    
    io = PDBIO()
    io.set_structure(mobile_struct)
    io.save(output_path)

def create_chimerax_script(output_dir, study_name, visualization_type, structures):
    """Create ChimeraX script for visualization."""
    script_path = os.path.join(output_dir, f"{study_name}_visualization.cxc")
    # logger.info(f"Creating ChimeraX script: {script_path}")
    
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
    pro_coords = torch.load(result["coords_path_pro"], weights_only=True)
    anti_coords = torch.load(result["coords_path_anti"], weights_only=True)
    pred_coords = torch.load(result["pred_coords_path"], weights_only=True)
    
    # logger.info("Coordinate shapes:")
    # logger.info(f"Pro: {pro_coords.shape}")
    # logger.info(f"Anti: {anti_coords.shape}")
    # logger.info(f"Pred: {pred_coords.shape}")
    
    logger.info(f"Model: {study_info['model']}")
    logger.info(f"Pred motif seq: {''.join([ch for (ch, mask) in zip(result['pred_sequence'], result['motif_mask']) if mask])}")
    logger.info(f"GT motif seq: {''.join([ch for (ch, mask) in zip(result['sequence_pro'], result['motif_mask']) if mask])}")

    # Convert to PDB
    merged_seq_pro = merge_seq(result["sequence_pro"], result["sequence_anti"])
    merged_seq_anti = merge_seq(result["sequence_anti"], result["sequence_pro"])
    sequence_to_pdb(merged_seq_pro, pro_coords, pro_pdb)
    sequence_to_pdb(merged_seq_anti, anti_coords, anti_pdb)
    sequence_to_pdb(result["pred_sequence"], pred_coords, pred_pdb)
    
    # Get scaffold positions
    scaffold_positions = get_scaffold_positions(result["motif_mask"], pro_coords, anti_coords)
    motif_start_idx = result["motif_mask"].index(1)
    motif_end_idx = len(result["motif_mask"]) - result["motif_mask"][-1::-1].index(1) - 1
    # logger.info(f"Motif start idx: {motif_start_idx + 1}, Motif end idx: {motif_end_idx + 1}")
    
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

def merge_seq(seq_1, seq_2):
    return "".join([char_1 if char_1 != "-" else char_2 for char_1, char_2 in zip(seq_1, seq_2)])

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
    pro_coords = torch.load(result_1["coords_path_pro"], weights_only=True)
    anti_coords = torch.load(result_1["coords_path_anti"], weights_only=True)
    pred_coords_pro = torch.load(result_1["pred_coords_path"], weights_only=True)
    pred_coords_anti = torch.load(result_2["pred_coords_path"], weights_only=True)
    
    # logger.info("Coordinate shapes:")
    # logger.info(f"Pro: {pro_coords.shape}")
    # logger.info(f"Anti: {anti_coords.shape}")
    # logger.info(f"Pred: {pred_coords_pro.shape}")
    # logger.info(f"Pred: {pred_coords_anti.shape}")
    motif_start_idx = result_1["motif_mask"].index(1)
    motif_end_idx = len(result_1["motif_mask"]) - result_1["motif_mask"][-1::-1].index(1) - 1
    # logger.info(f"Motif start idx: {motif_start_idx + 1}, Motif end idx: {motif_end_idx + 1}")
    motif_start_idx = result_2["motif_mask"].index(1)
    motif_end_idx = len(result_2["motif_mask"]) - result_2["motif_mask"][-1::-1].index(1) - 1
    # logger.info(f"Motif start idx: {motif_start_idx + 1}, Motif end idx: {motif_end_idx + 1}")
    
    # Convert to PDB
    merged_seq_pro = merge_seq(result_1["sequence_pro"], result_1["sequence_anti"])
    merged_seq_anti = merge_seq(result_1["sequence_anti"], result_1["sequence_pro"])
    sequence_to_pdb(merged_seq_pro, pro_coords, pro_pdb)
    sequence_to_pdb(merged_seq_anti, anti_coords, anti_pdb)
    sequence_to_pdb(result_1["pred_sequence"], pred_coords_pro, pred_pdb_pro)
    sequence_to_pdb(result_2["pred_sequence"], pred_coords_anti, pred_pdb_anti)
    logger.info(f"Model: {study_info['model']}")
    logger.info(f"Pred motif pro seq: {''.join([ch for (ch, mask) in zip(result_1['pred_sequence'], result_1['motif_mask']) if mask])}")
    logger.info(f"Pred motif anti seq: {''.join([ch for (ch, mask) in zip(result_2['pred_sequence'], result_2['motif_mask']) if mask])}")
    logger.info(f"GT motif seq: {''.join([ch for (ch, mask) in zip(result_1['sequence_pro'], result_1['motif_mask']) if mask])}")
    
    # Get scaffold positions
    scaffold_positions = get_scaffold_positions(result_1["motif_mask"], pro_coords, anti_coords)
    
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
            "domain_pro": "4mbpA01",
            "domain_anti": "1ezpA02"
        },
        {
            "name": "best_individual_protein_mpnn",
            "model": "protein_mpnn",
            "domain_pro": "4mbpA01",
            "domain_anti": "1ezpA02"
        },
    ]

    studies_2 = [
        {
            "name": "best_overall_cs",
            "model": "cs_design",
            "domain_pro": "2odhA01",
            "domain_anti": "3imbB01"
        },
        {
            "name": "best_overall_mpnn",
            "model": "protein_mpnn",
            "domain_pro": "3uoaC02",
            "domain_anti": "6f7iB02"
        },
    ]
    
    algorithm_comparisons_rmsd_diff = [
        {
            "name": "largest_algorithm_preference_diff_cs_design",
            "domain_pro": "5mw1B02",
            "domain_anti": "4cj7B02",
        },
        {
            "name": "largest_algorithm_preference_diff_protein_mpnn",
            "domain_pro": "4cj7B02",
            "domain_anti": "5mw1B02",
        },
    ]

    algorithm_comparisons_summed_rmsd = [
        {
            "name": "largest_algorithm_diff_summed_rmsd_cs_design",
            "domain_pro": "4c8rF01",
            "domain_anti": "4bgkA01",
        },
        {
            "name": "largest_algorithm_diff_summed_rmsd_protein_mpnn",
            "domain_pro": "3a77C00",
            "domain_anti": "5jejA00",
        },
    ]
    
    base_output_dir = "visualization_output"
    
    # Process individual studies -> keep
    for study in studies:
        study_dir = os.path.join(args.output_dir, base_output_dir, study["name"])
        process_study(results, study, study_dir)

    for study in studies_2: # -> keep
        study_dir = os.path.join(args.output_dir, base_output_dir, study["name"])
        process_study_2(results, study, study_dir)

    # Process algorithm comparisons -> keep
    for study in algorithm_comparisons_rmsd_diff:
        study_dir = os.path.join(args.output_dir, base_output_dir, study["name"])
        for model in ["cs_design", "protein_mpnn"]:            
            model_dir = os.path.join(study_dir, model)            
            study["name"] += f"_{model}"
            study["model"] = model
            process_study(results, study, model_dir)

    # Process algorithm comparisons
    for study in algorithm_comparisons_summed_rmsd:
        study_dir = os.path.join(args.output_dir, base_output_dir, study["name"])
        for model in ["cs_design", "protein_mpnn"]:            
            model_dir = os.path.join(study_dir, model)            
            study["name"] += f"_{model}"
            study["model"] = model
            process_study_2(results, study, model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/jastern33/code/bayes_design_data")
    args = parser.parse_args()
    main(args)