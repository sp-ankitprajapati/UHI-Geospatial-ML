# Updated UHI interview questions aur coding drills

Earlier pack ke 50 fundamentals ke saath yeh 20 updated questions practice karo. Answers pehle khud bolo, phir compare karo.

| Question | Answer ka core |
| --- | --- |
| 1. Upgrade mein kya improve hua? | Reproducibility, band contracts, QA, common grid, paired stats, spatial evaluation, tests aur evidence. Accuracy increase claim nahi. |
| 2. Spatial score itna low kyun? | Labels few clustered locations se hain. Held-out blocks mein classes training se absent hain. Different geography par generalization weak hai. |
| 3. 2016 model water kyun nahi predict kar sakta? | All 41 labeled water points QA-rejected. Retained training mein class 1 absent. Independent QA/label audit aur new clear examples chahiye. |
| 4. Random accuracy 100% aur macro F1 .8 kaise? | Fixed five-class averaging mein absent water F1 zero, remaining four perfect. |
| 5. Old CSV actual features kaise verify kiye? | Source point coordinates par raw bands sample kiye; original CSV feature values se exact DN equality check ki. |
| 6. Temperature feature RF se kyun hataya? | Model optical land cover predict karta hai. Thermal feature later temperature association ko circular bana sakta tha. |
| 7. QA_RADSAT zero bug kya tha? | Zero means valid. Destination NoData zero rakhne par warper valid values badal sakta tha. Separate sentinel aur shifted-grid regression test use kiya. |
| 8. 300 m buffer group split se extra kya karta hai? | Adjacent blocks ke close train/test neighbors remove karta hai. Group boundary alone distance separation guarantee nahi. |
| 9. 1 km aur 300 m scientifically optimal hain? | Nahi, explicit evaluation design choices hain. Future deployment aur spatial autocorrelation analysis se justify/sensitivity test karna hai. |
| 10. Missing class wali folds hide kyun nahi ki? | Yeh data coverage ka core limitation hai. Easy folds select karna model quality overstate karega. |
| 11. Mean delta +7.40 ko CV mein kaise bolo? | Two selected dates par 591,749 paired clear pixels ka observed mean LST difference. Long-term climate warming nahi. |
| 12. Original +8.2 kya hua? | Original clipped minima actually 21.06/29.47 mile. New QA/paired support aur minima different definitions hain; original claim as stated retain nahi kiya. |
| 13. 145 wards ka matlab Hyderabad mein sirf 145 wards? | Nahi. Supplied file feature count145 hai. Complete municipal inventory independently verify nahi hui. |
| 14. True UHI intensity calculate hui? | Defined rural comparator nahi, isliye current output surface-temperature comparison hai. |
| 15. Pearson r p-value kyun nahi diya? | Neighboring wards dependent ho sakte hain aur class maps weak hain. Naive IID significance/causal ranking support nahi. |
| 16. Full pipeline memory bounded hai? | AOI arrays memory mein hain. Prediction chunks bounded hain. Whole-scene window streaming future extension hai. |
| 17. SHA checksum kya protect karta hai? | Same input bytes verify karta hai. Scientific validity ya label correctness prove nahi karta. |
| 18. Unit tests se model valid ho gaya? | Implementation behavior check hua. Independent representative labels aur deployment evaluation phir bhi chahiye. |
| 19. NLP/LLM kahan use hua? | Is upgraded UHI pipeline mein nahi. Future grounded report search useful ho sakta hai; temperature engine ko LLM ki need nahi. |
| 20. AI ne code banaya toh tumhara contribution? | AI-assisted implementation honestly bolo. Jo parts khud samjhe, verify kiye, tests/debugging/data audit kiya woh examples do. Independent practice ke bina expertise claim mat karo. |

## Coding drill A: array aur units

Input `dn=[0,45000,46000]`, validity `[True,True,False]`. Function output `[nan,29.6609,nan]` hona chahiye. Shape mismatch par clear error do. Follow-up: float32 vs float64, units, zero fill aur offset explain karo.

## Coding drill B: shape restoration

Mask `[[True,False],[True,True]]`, labels `[2,3,5]`. Output `[[2,0],[3,5]]`. Labels count mismatch par error. Follow-up: `cube[:,mask].T` ko step-by-step explain karo aur invalid feature nan ko kaise handle karoge?

## Coding drill C: spatial neighbors

Points x=[999,1001,1800], y=0. First point test hai. 1 km blocks first aur second ko different groups denge, lekin 300 m buffer second ko training se remove karega. Third retain hoga. KD-tree aur direct distance matrix ke time/memory tradeoffs discuss karo.

## Coding drill D: paired statistics

Old=[10,100], new=[12,nan]. Independent means aur paired means compare karo. Ab two wards aur coverage fraction add karo. Follow-up: empty ward, all-NaN ward, mean-vs-median aur area weighting.

## Coding drill E: evaluation evidence

Published prediction CSV load karo. `actual==predicted` ka mean nikalo. Confusion matrix banao. Group test rows ko fold by fold count karo aur ensure each sample once test hua. Train/test group intersection empty prove karo. Follow-up: same geographic point multiple years mein ho toh pooled model ke groups kaise banenge?

## Mock interview ka format

First round: 90-second intro, problem/data/output. Second: randomly chosen module ka input/output/error case. Third: upar ka coding drill without copying. Fourth: weak result defend karo, metric invent kiye bina next experiment propose karo. Jab tum teen alag sessions mein yeh independently kar pao, project ki interview readiness ka practical evidence hoga.
