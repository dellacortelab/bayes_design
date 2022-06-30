
def compare_forward_prob(args):                                                        
                                                                                 
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")    
                                                                                 
    model_1 = model_dict['protein_mpnn'](device=device) 
    model_2 = model_dict['bayes_struct'](device=device)                      
                                                                                 
    # Get sequence and structure of protein to redesign                          
    seq, struct = get_protein(args.protein_id)                                   
                                                                                 
    # Masked positions are the positions to predict/design                       
    # Default to predict all positions                                           
    masked_positions = np.ones(len(seq))                                         
    # Preserve fixed positions                                                   
    #for i in range(len(args.fixed_positions), step=2):                           
    #    fixed_range_start = args.fixed_positions[i]                              
    #    fixed_range_end = args.fixed_positions[i+1]                              
    #    masked_positions[fixed_range_start:fixed_range_end] = 0.                 
    masked_seq = ''.join(['-' if mask else char for char, mask in zip(seq, masked_positions)])
                                                                                 
    # Decode order defines the order in which the masked positions are predicted 
    decode_order = decode_order_dict['proximity_decode_order'](masked_seq)              
                                                                                 
    # The decoding algorithm determines how the sequence is decoded              
    designed_seq_1 = decode_algorithm_dict['beam'](prob_model=model_1, struct=struct, seq=masked_seq, decode_order=decode_order, n_beams=16)
    designed_seq_2 = decode_algorithm_dict['beam'](prob_model=model_2, struct=struct, seq=masked_seq, decode_order=decode_order, n_beams=16)
    
    forward_struct_model = TrRosettaWrapper()
    distogram_orig = forward_struct_model(seq)
    distogram_1 = forward_struct_model(designed_seq_1)
    distogram_2 = forward_struct_model(designed_seq_2)
    
    correct_bins = torch.argmax(distogram_org, dim=-1)
    log_prob_orig = torch.sum(torch.log(distogram_orig[:, :, correct_bins]))
    log_prob_1 = torch.sum(torch.log(distogram_1[:, :, correct_bins]))
    log_prob_2 = torch.sum(torch.log(distogram_2[:, :, correct_bins]))
    print("Original Sequence", seq)
    print("Prob:", prob_orig)
    print("Designed Sequence ProteinMPNN", designed_seq_1)
    print("Prob:", prob_1)
    print("Designed Sequence BayesDesign", designed_seq_2)
    print("Prob:", prob_2)

    return seq, masked_seq, designed_seq  


 Experiment: 
Take a structure with a known sequence. Use p(seq|struct) and p(seq|struct)/p(seq) 
to design new sequences for the structure. Compare the log-probability and stability
of the original structure and the two designed structures. In the best case, we
will see, from most stable to least stable: p(struct|seq) = p(seq|struct)/p(seq) > orig_seq > p(seq|struct)

Experiment:
Same as previous experiment, but hold part of the sequence fixed.

Experiment:
Compare perplexity and recapitulation performance to rmsd and log_prob performance
for p(seq|struct)/p(seq) and p(seq|struct). We expect p(seq|struct) to outperform
p(seq|struct)/p(seq) for perplexity and recapitulation, but expect p(seq|struct)/p(seq)
to outperform p(seq|struct) in rmsd and log_prob. We argue that perplexity and
recapitulation are the wrong metrics for sequence design.
