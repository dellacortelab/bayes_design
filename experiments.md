# Experiments

## Designed proteins

### WN Model System
Original Sequence:
```
KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```
Original sequence with mask, preserving fixed residues:
```
---------------------------------G
```
Sequence Stats:
- BayesDesign LogP: -89.15804301725373 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG`
- ProteinMPNN LogP: -64.23751689474135 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG`

#### BayesDesign
Command:
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34
```
Sequence:
```
MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG
```
Sequence Stats:
- BayesDesign LogP: -31.573002420457204 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG`
- ProteinMPNN LogP: -52.71265054032396 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG`


#### ProteinMPNN
Command:
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34
```
Sequence:
```
TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG
```
Sequence Stats:
- BayesDesign LogP: -73.07880518446302 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name bayes_design --decode_order n_to_c --fixed_positions 34 34 --sequences TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG`
- ProteinMPNN LogP: -30.726229193957487 `python3 experiment.py compare_seq_probs --protein_id 1PIN --model_name protein_mpnn --decode_order n_to_c --fixed_positions 34 34 --sequences TLPEGWVERVDPKTGEKVFFNTKTKEVTKEKPVG`

### Plastic Degrading Enzyme
Original Sequence:
```
TDPGNGSGYQRGPDPTVSFLEAARGQYTVDTERVSSLVGGFGGGTIHYPEDVSGTMAAIVVIPGYVSAESSIEWWGPKLASYGFVVMTIDTNTGFDQPPSRATQINAALDYLVDQNSDNGSPVQGMIDTSRLGVIGWSMGGGGTIRVASQGRIKAAIPLAPWDTSSYYARRAEAATMIFACESDVVAPVGLHASPFYNALPSSIDKAFVEINNGSHFCANGGGINNDVLGRLGVSWMKLHLDEDGRYNQFLCGPNHESDFSISEYRGNCPYGSHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP:
- ProteinMPNN LogP: 

Original sequence with mask, preserving fixed residues:
```
----------------------------------------------------------------YVS-----------------------------Q---------------------------------------WSM----------------------W------------------C--D--------------------------IN--SHFC--GGGINN-------------------------C--------S-------C----HHHHHHHH
```

#### BayesDesign
Command:
```
python3 design.py --model_name bayes_design --protein_id plastic_degrading_enzyme --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 138 138 184 184 216 216 65 65 97 97 139 139 162 162 66 67 137 137 211 212 215 215 217 217 223 225 261 261 221 226 181 181 218 218 252 252 269 269 274 281
```
Sequence:
```
AIQWGKDAHVRGPIPTKELLCADHGFWEVSTQDIPASVKGFGGWTIHFPQNWEGKLPAVVCMPGYVSNADSIAWWGPRLASFGFVVAVINHTRPDCQPEEMAQEIKAAMDHLIAQNKDPQSPIHGKIDENRLGVIGWSMGGGATIIVASDGRVKAAIPLCPWLNSTEPAKKATADVLIFSCENDTVCPPEKHSIPMWEAIPKEIDRMHVMINNGSHFCATGGGINNCTLNLLAVSWLRLHLMNDQSVEKFLCGPEIDNDPSISKKESNCPFGQHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP: -212.90735622198204
- ProteinMPNN LogP: 

#### ProteinMPNN
Command:
```
python3 design.py --model_name protein_mpnn --protein_id plastic_degrading_enzyme --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 138 138 184 184 216 216 65 65 97 97 139 139 162 162 66 67 137 137 211 212 215 215 217 217 223 225 261 261 221 226 181 181 218 218 252 252 269 269 274 281
```
Sequence:
```
AVAPGGHPLVRGPAPTAALLAAPAGPWAVATEAIPASVAGFGGGTVHFPLDVEGRLPAVVIAPGYVSSADSIAWWGPALASHGFVVLVINWTSPTVQPAEGAAEIEAALAHLDAANDDPASPIRGLIDRDRRGVIGWSMGGGATVIVASTGRVRAAIPLAPWLPSTEPAARATAATLIISCENDTVTPPEKFSIPIFEALPPEIDRMLVQINNGSHFCATGGGINNATLRRLVIAWLRLHLMDDPRVEAFLCGPAIESDPSISRFESNCPFGRHHHHHHHH
```
Sequence Stats:
- BayesDesign LogP:
- ProteinMPNN LogP: -180.78068155687816

### Recently Evolved Rice Protein (dn47)

Original Sequence:
```
STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI  STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI
```
Original Sequence with mask, preserving fixed residues:
```
--------------------------------------------------------------------------------------------------------------------------------------------
```
Sequence Stats:
- BayesDesign LogP: -944.9901678422955 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAISTASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`
- ProteinMPNN LogP: -268.70771463316356 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAISTASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`

Hypotheses:
- BayesDesign increases specificity
    - Redesign protein for which the WT takes multiple conformations
        - SA + JS Try to increase specificity of active enzyme conformation, test enzymatic activity
            - Kinase - stabilize active conformation
                - SA - Find kinase with active conformation, send to JS
                - JS - Redesign to prefer active conformation
        - Test alpha-beta conformation protein with CD
        - SAX to see % of redesigned protein in each conformation
    - Amyloid beta sheets

- By increasing specificity, we increase stability (especially if unfolded state is included in the energy landscape)
    - Redesign ligand protein (nanobody, HIV protein) to bind with more affinity
        - Measure k_d
    - AM + JS - CD at different temperatures or differential scanning fluorometry
        - PETase
        - dn_47
        - HIV protein (JS redesign)
    - AM - Measure change in enzyme activitity
        - PETase

- BayesDesign helps avoid ProteinMPNN repetition problem for de novo proteins
    - JG + JS - Redesign de novo rice proteins
    - JG + JS - Redesign random protein (g c bias)
        - Find other weird ProteinMPNN cases
        - Show that AlphaFold is also unexpectedly confident on those cases

- Unconstrained backbone helps us design even more specific structures
    - JS - work on unconstrained backbone case
    
    

maximum p(seq|struct) / p(seq) = maximum log p(seq|struct) - log p(seq)

#### BayesDesign
Command:
```
python3 design.py --model_name bayes_design --protein_id dn47 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM
```
Sequence Stats:
- BayesDesign LogP: -163.4057692585695 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM`
- ProteinMPNN LogP: -165.74089450285308 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences LNEHMAISVESLKQKWCEAHEQAVQDAFTRPEGWDIPEQVWQARANKAACLSCQKVNEVMDKLIPPESERKFKNDPLTVDKLCDACCQAVCWAVQEALSEPPGVDWPREQQDHIAQKSAQKTCEGTRAVMNECIEPPKIM`


#### ProteinMPNN
Command:
```
python3 design.py --model_name protein_mpnn --protein_id dn47 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK
```
Sequence Stats:
- BayesDesign = ProteinMPNN / XLNet LogP: -578.6982698784458 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name bayes_design --decode_order n_to_c --sequences PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK`
- ProteinMPNN LogP: -101.24846088136692 `python3 experiment.py compare_seq_probs --protein_id dn47 --model_name protein_mpnn --decode_order n_to_c --sequences PPPKKKPSKEEIIKKVVEAVKKAVKEAFKKPEGLDIPEEEWEKIAEKAAEKAAKKVEEVLKKEIPPEEEKKKKKKKLTKEKIIEKIAKAVKEAVKEALKEPEGLDIPEELAKKIAEEAAKKAKKEVEKVLKEVIPKKKKK`

### Recently Evolved Rice Protein (dn47_cut2)

Original Sequence:
```
STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI
```
Sequence Stats:
- BayesDesign LogP: -867.6463986468442 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`
- ProteinMPNN LogP: -245.032467840132 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences STASSSLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPLSHDALVDTITTAVADAIRISYQSPPGLTVDDAVTKSIADSAADKASAAVRDALSKDLPSPIAI`

#### BayesDesign
Command:
```
python3 design.py --model_name bayes_design --protein_id dn47_cut2 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI
```
Sequence Stats:
- BayesDesign LogP: -142.98842013604815 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI`
- ProteinMPNN LogP: -222.6793760330641 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences ECSMHMCTEDCFIDQVVLAVSQAIKQWFEPESGQAIPREQINAHADKCAQMGSQAVRDVLKEKLPCNVSPEEFIELVVKAHSQACLQALTMPEGQAWPEFQRNEHANGAAKGVSQACKCVIDECIPKPPDI`


#### ProteinMPNN
Command:
```
python3 design.py --model_name protein_mpnn --protein_id dn47_cut2 --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128
```
Sequence:
```
PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA
```
Sequence Stats:
- BayesDesign LogP: -501.2846077182429 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name bayes_design --decode_order n_to_c --sequences PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA`
- ProteinMPNN LogP: -91.01211402479034 `python3 experiment.py compare_seq_probs --protein_id dn47_cut2 --model_name protein_mpnn --decode_order n_to_c --sequences PPPPAAPTPEELREAVARAVREAVREVFRPPPGDPTPPELWERLAEEAARAAEEAVRRVLEERLPEPLPPEELREAIARAVREAVRRVLEGEPGDPIPEELRERLAEEAARAAREAVEEVLRRLLPPPPPA`

## Other experiments

### Visualizing distribution shift
```
python3 experiment.py viz_probs --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --sequence KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG --results_path ./results/probs_viz_KLPP
```
```
python3 experiment.py viz_probs --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --sequence KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG --results_path ./results/probs_viz_KLPP_bayes_002 --bayes_balance_factor .002
```
### Evaluate perplexity
```
python3 experiment.py compare_seq_metric --protein_id 1PIN --fixed_positions 34 34 --model_name protein_mpnn --metric perplexity --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```

# Compare redesign design, from scratch design, wild type
```
python3 experiment.py compare_seq_metric --protein_id 1PIN --decode_order n_to_c --model_name bayes_design --metric log_prob --sequences QLPEGWEEKVDEETKEKIYYNKETKEITKEKMIC MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```

One problem with using mlm redesign: Even if the probability increases for one token, that probability could be changed once we replace other tokens. So maybe V instead of K increases probability by .01, and then G instead of L increases probability by .01, but decreases probability of V by .2. Perhaps we should instead choose each token based on how it affects the log-likelihood of the whole sequence instead of just that token. argmax_aa p(struct=X|seq=seq:seq[i]=aa) vs argmax_aa p(struct=X|seq)

This:   p(seq:seq[i]=aa|struct)/p(seq:seq[i]=aa)
        = p(seq[i]=aa|struct, seq[!=i]=seq) p(seq[!=i]=seq|struct) /
           ( p(seq[i]=aa|seq[!=i]=seq) p(seq[!=i]=seq) )

Not this: p(seq[i]=aa|struct, seq[!=i]=seq)/p(seq[i]=aa|seq[!=i]=seq)

# Price sequences - 1PIN (beta sheet) aka WW

From scratch (bidirectional autoregressive) log probs:
original sequence               KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG  -111.97058678054292

BayesDesign
greedy design from scratch,     TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG  -40.06295947692667
beam 128 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724
beam 256 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724
beam 512 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG  -31.762047139937724

ProteinMPNN
greedy design from scratch,     TLPEGWVEVVDPETGEKKYYNTKTKEVTSEKPVG  
beam 128 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 256 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 512 design from scratch,   KLPEGWVEKVDPKTGEKVYYNTKTKEITKEKPVG  


# Price sequences 1COI (coiled coil) aka 1CW
From scratch (bidirectional autoregressive) log probs:
original sequence               KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG

BayesDesign
greedy design from scratch,     TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG
beam 128 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG
beam 256 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG
beam 512 design from scratch,   QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG

ProteinMPNN
greedy design from scratch,     TLPEGWVEVVDPETGEKKYYNTKTKEVTSEKPVG
beam 128 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 256 design from scratch,   KLPEGWVEKVDPKTGKKVYYNTKTKEITEEKPVG  
beam 512 design from scratch,   KLPEGWVEKVDPKTGEKVYYNTKTKEITKEKPVG  

## Price sequence design commands
Greedy design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```
Greedy redesign BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm greedy
```
Beam 128 redesign BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 128
```
Beam 256 redesign BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 256
```
Beam 512 redesign BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512
```
Greedy design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34   --decode_algorithm beam_medium --n_beams 512 --from_scratch
```
Greedy redesign ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm greedy
```
Beam 128 redesign ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34  --decode_algorithm beam_medium --n_beams 128
```
Beam 256 redesign ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --decode_algorithm beam_medium --n_beams 256
```
Beam 512 redesign ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --decode_algorithm beam_medium --n_beams 512
```

1CW

Greedy design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

Greedy design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29  --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id 1coi --decode_order n_to_c --fixed_positions 29 29   --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

## Price sequence evaluation commands
python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --metric log_prob --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG QTWYGWVPIVDDKTGETKWLNKIEKKVTSKKPIG QWWYGWVPIVDEKTGEEKAYNVLTKEVTSERPIG

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --fixed_positions 34 34 --bayes_balance_factor .002 --metric log_prob --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG TLPEHWVKRKDPKTGQWIYENTKTHETLAQKWQG KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG QLPEGWVKRTNKVTGKDEYRNVKTNETTSKKPIG KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG QTWYGWVPIVDDKTGETKWLNKIEKKVTSKKPIG QWWYGWVPIVDEKTGEEKAYNVLTKEVTSERPIG --from_scratch

# Bundy sequences
Fix all residues except 16 engineered residues, greedy decode redesign
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 13 15 20 22 27 29 36 38 42 44 53 55 63 65 77 79 81 83 84 86 99 101 124 126  133 135 147 149 175 177 179
```
Fix the active site (small), greedy decode redesign
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160
```
Fix the active site (small), beam search decode from scratch
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions  1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104
```
Fix the active site (small) and the engineered residues, greedy decode redesign
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176
```
Fix the active site (small) and the engineered residues, beam search decode from scratch
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions  1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176
```
Fix the active site (large), greedy decode redesign
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 
```
Fix the active site (large), beam search decode from scratch
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162
```
Fix the active site (large) and the engineered residues, greedy decode redesign
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm greedy --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176 
```
Fix the active site (large), and the engineered residues, beam search decode from scratch
```
python3 design.py --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --decode_algorithm beam_medium --n_beams 512 --from_scratch --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176 
```

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRQTAGYNMDQVLEQGGASSLFQKLGVSVTPIQRIVLSGENGLKVDIHVIIPYEGLSGCQMGLIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYTGIAVFDGKKITVTGTLWNGNKIVDERLINPDGSLLFRVTINGVTGWRLCERILA --fixed_positions 1 9 10 13 15 20 22 27 29 36 38 42 44 53 55 63 65 77 79 81 83 84 86 99 101 124 126 133 135 147 149 175 177 179

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLDDFVGKWELVEQKNIPEVLRQMGAPQFFIDLWCNTKPILVITKYGENGLKVTIEMVIPKKGLTCDQMDQIHKIFKVMIPVDENHFKVILDYGTLIINGVSPNCKDWLGRPYEGICTFDGKKITVTGTLPNGNKFIDTYEILPDGSLLFTVDVNGVKGWWKLKRVEE --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLDDFVGNWRMVSQWNIPAVLREMGMPPFLIDLWCATTPIWVITKYGENGLKVDVHMVIPKEGLTPEQMRYLQAMFGHMTQVDETHFQVILDYGVFIINGTSKNCKDFMNRPFEVNTTFDGKKLTMTGTLWNGKKFVMTFEILPDGHLRYTVDVNGVKGWMILERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLTLEDFVGDWRLVDKWNLPEVLKAMGVPQFMINLYCQTQPILRITKAGENGLKIEIEMVIPKKGLTCDQMEQIKKIYKHVEDVDDNHFKVILDYGTLIINGVSPNMKDFLGRPYEGICTFDGKKITVTGTLPNGNKVIITFEIQPDGSLLLTIDVNGVKGWMVYERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVLKLEDFVGDWRRVDSWNLPEVLKAMGVPQFFINLFCQTQPIWRISKHGEKGLKIQMIMRIPKQGLTPDQMAQIQKTFKHVQDIDDQHFQVILDYGTLIIDGVSPNCKDFLGRPYEGICKFDGKKITVTGTLPNGNKFIWTMEILDDGSLLFTVDVNGVKGYMILERVEP --fixed_positions 1 9 10 10 13 13 51 51 59 59 61 61 63 63 100 102 104 104 106 106 112 112 114 114 121 125 127 127 132 132 134 137 144 144 146 146 148 148 156 156 158 158 160 160 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGKWRMVSKQNTPAVLKEEGAPQFLIDLWCNTTPIFIITLSGENGLKIDIEMIIPKKGLTCDQMKYLQKIFKVMIPVDENNFKVILHYGTLVIDGVTPNMKDYFGRPYEGICKFDGKKITVTGTLWNGNKIIDEWEILPDGSLLFRRTVNGVVGWWKLERIEE --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRLVSKQNMAAVLREMGAPDFLIQLYLQCTPIFHITKSGENGLKIDVEMIIPKAGLTPEQMCYLQKMFKHMEPVDENHFKVILHYGTLVIDGVTPNMKDAFGRPYEGICKFDGKKITVTGTLWNGNKIIDEYEILPDGSLLFRRTVNGVTGWMKLERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWRMVKQWNLPAVLKAMGVPQFMINLFCQTTPILRITLSGENGLKIDIEMIIPKKGLTCDQMNQIKKIFKHVEDVDDNNFKVILHYGTLVIDGVTPNMKDWFGRPYEGICKFDGKKITVTGTLWNGNKIIDEFEILPDGSLLFRVTVNGVEGWMIYERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

python3 experiment.py compare_seq_metric --model_name bayes_design --protein_id nanoluc --decode_order proximity --bayes_balance_factor .002 --metric log_prob --from_scratch --sequences MWSHPQFEKVFTLEDFVGDWRQTAGYNLDQVLEQGGVSSLFQNLGVSVTPIQRIVLSGENGLKIDIHVIIPYEGLSGDQMGQIEKIFKVVYPVDDHHFKVILHYGTLVIDGVTPNMIDYFGRPYEGIAVFDGKKITVTGTLWNGNKIIDERLINPDGSLLFRVTINGVTGWRLCERILA MWSHPQFEKVFTLEDFVGDWREVDRWNLADVLKAMGVPQFLINLYMSCTPIWRITKSGENGLKIDVEMIIPKQGLTEDQLQQIKKIFQHVEDVDDNHFKVILHYGTLVIDGVTPNMKDWFGRPYEGICKFDGKKITVTGTLWNGNKIIDEFEILPDGSLLFRTTVNGVTGYRILERVEP --fixed_positions 1 9 8 14 17 17 49 51 57 64 98 108 110 116 120 127 132 139 142 150 156 158 160 162 14 14 21 21 28 28 37 37 43 43 53 54 64 64 78 78 82 82 85 85 100 100 125 125 134 134 148 148 176 176

# Moody lab design

# Moody sequences
From scratch (bidirectional autoregressive) log probs:
original sequence               SATHIKFSKRDEDGKELAGATMELRDSSGKTISTWISDGQVKDFYLYPGKYTFVETAAPDGYEVATAITFGHHHHHHHHHH

BayesDesign
greedy design from scratch,     MPHEVRYEKRDSNGRLVKGYNWSLLDAEFNVIATWVSDGKPKYFMLPDGIYTWVETQAPPGYPKQPDVTFGHHHHHHHHHH
beam 128 design from scratch,   GKQHVEIRKVDKNGRLLKGAKWELLNSEGDVIDQWISDGRPKHFWLPWGHYTLRETEPMPGFPKRPPETFGHHHHHHHHHH
beam 256 design from scratch,   
beam 512 design from scratch,   

ProteinMPNN                     
greedy design from scratch,     SLEKVTIKKVDSNGNLLSGAKWELLDENGNVIKTWTSDGKPKEFELPPGIYTVKETEAPAGYSKAADETFGHHHHHHHHHH
beam 128 design from scratch,   EKKKVTIEKKDENGNLLKGAKWELLNEKGEVIEEWTSDGKPKEFELPEGIYTVKETEAPEGYEKKEPETFGHHHHHHHHHH
beam 256 design from scratch,     
beam 512 design from scratch,   
  
Greedy design from scratch BayesDesign
```[]
python3 design.py --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch BayesDesign
```
python3 design.py --model_name bayes_design --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --bayes_balance_factor .002  --decode_algorithm beam_medium --n_beams 512 --from_scratch
```

Greedy design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm greedy --from_scratch
```
Beam 128 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 128 --from_scratch
```
Beam 256 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 256 --from_scratch
```
Beam 512 design from scratch ProteinMPNN
```
python3 design.py --model_name protein_mpnn --protein_id spycatcher --decode_order n_to_c --fixed_positions 9 9 55 55 72 81 --decode_algorithm beam_medium --n_beams 512 --from_scratch
```