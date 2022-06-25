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
