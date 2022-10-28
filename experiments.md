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

python3 design.py --model_name bayes_design --protein_id 5ibo --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34

python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --decode_algorithm greedy --fixed_positions 34 34 --bayes_balance_factor .002

python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34 --bayes_balance_factor .002

python3 design.py --model_name bayes_design --protein_id 1PIN --decode_order n_to_c --decode_algorithm beam_medium --n_beams 128 --fixed_positions 34 34 --bayes_balance_factor .002 --from_scratch

# Compare redesign design, from scratch design, wild type
```
python3 experiment.py compare_seq_metric --protein_id 1PIN --decode_order n_to_c --model_name bayes_design --metric log_prob --sequences QLPEGWEEKVDEETKEKIYYNKETKEITKEKMIC MLPEGWVKQRNPITGEDVCFNTLTHEMTKFEPQG KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG
```
greedy design from scratch,     VLPEIWKRQINPETNQEQYWNTKTHTTTKQKPQG
greedy design redesign,         KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG
beam design from scratch,       VLPQGWKQRKSPKTNKTIYENTITKTITSKKPIG
beam design redesign,           KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG


# greedy redesign, greedy design + bayes, beam redesign, beam redesign + bayes, old beam from scratch, 
python3 experiment.py compare_seq_metric --protein_id 1PIN --decode_order n_to_c --model_name bayes_design --metric log_prob --sequences KLPPGWEKRMSRSSGRVYYFNHITNASQFERPSG VLPEIWKRQINPETNQEQYWNTKTHTTTKQKPQG KTPEWWWPIINKWTMETMYYNTGTNEVTKEKPIG VLPQGWKQRKSPKTNKTIYENTITKTITSKKPIG KTWYGWVPIVDFKTGEEMYRNDLTNEITRDKPIG