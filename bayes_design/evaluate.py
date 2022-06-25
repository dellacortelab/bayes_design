
"""
Evaluation for Seq Design Models that maximize p(struct=S|seq) - i.e. sequence
design models made to design a sequence that matches a backbone.

Each metric should demonstrate that p(seq|struct)/p(seq) is better than p(seq|struct).
"""


def evaluate_log_prob(sequence, seq_to_struct_model, targ_struct):
    """
    Evaluate the log probability of the structure for the sequence under a 
    sequence to structure model. This measures p(struct|seq) for a designed sequence.
    We expect p(seq|struct)/p(seq) to be better than p(seq|struct) because 
    p(seq|struct)/p(seq) seeks to model p(struct|seq). This may be a flawed metric
    because the seq-to-struct network may only key in on a few important residues 
    for global structure, possibly failing to correctly model the probabilities of
    residues that are less important for folding.

    You should call evaluate_log_prob with a sequence designed with p(seq|struct)
    and with a sequence designed with p(seq|struct)/p(seq) and compare the log
    probabilities.
    
    Args:
        sequence (str): a string (no spaces) representing an amino acid sequence
        seq_to_struct_model (nn.Module): a network taking as input a 
    """
    

def evaluate_rmsd(sequence, seq_to_struct_model, targ_struct):
    """
    Evaluate RMSD between the "true structure" (computationally predicted from 
    the true sequence) and the predicted structure for a designed sequence. This 
    is a good metric for evaluating p(struct|seq), but has a performance ceiling.
    You can only match the RMSD. Ideally, we want to design a sequence with RMSD
    as close to 0 as possible, but with higher stability than the original sequence
    (measured by evaluate_log_prob).
    """

def evaluate_perplexity(orig_seq, seq_model, struct_to_seq_model):
    """
    Evaluate the perplexity of the original sequence under p(seq|struct)/p(seq) vs.
    p(seq|struct). We expect the perplexity to be higher for p(seq|struct), because
    perplexity is the training objective of the p(seq|struct) model.
    """
    

def evaluate_recapitulation(pred_seq, orig_seq):
    """
    Evaluate % recapitulation of the original sequence. This is a bad metric for 
    measuring p(struct|seq) because it really measures p(seq|struct).
    """
    perc_recapitulate = (predict_seq == orig_seq).mean()
    return perc_recapitulate

def evaluate_stability():
    """
    This is the best metric, as it measures p(struct|seq), without being subject
    to the approximations of using a computational model for p(struct|seq). But it
    must be based on experimental results.
    """
    pass
