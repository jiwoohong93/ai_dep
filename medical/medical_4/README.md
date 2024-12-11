# Public Data Assisted Differentially Private In-Context Learning
## This is repository of project "Public Data Assisted Differentially Private In-Context Learning"  
--- 
## Overview

This work aims to achieve private in-context learing robust to inference attack while maintaining utlity with the help of public data 

<img src="./Figure/Schematic_diagram.PNG" width="1000px" title="Framework"></img>

1. First, in-context examples are sampled from both private and public datasets (Step 1).  
2. Using these in-context examples, generate LLM responses (Step 2-1).  
3. Privately aggregate the LLM responses and select a few candidates (Step 2-2).  
4. Finally, choose the best response, guided by the public example (Step 3).  



## How to use 
1. Install requirments

```bash
pip install -r requirements.txt 
```

2. Run in-context learning with each ICL example 

```bash
run.sh 
```

3. Construct semantic group 

```bash
semantic_group.sh 
``` 

4. Generate final response running `{benchmark}.ipynb` (e.g., `ChatDoctor.ipynb`) 

## Evaluation result 
### ChatDoctor Benchmark 
| **Method**         | **Metrics**    | **ε=1**           | **ε=3**           | **ε=8**           | **ε=∞ (Agg)**    | **ε=0 (0-shot)** | **ε=∞ (4-shot)** |
|---------------------|----------------|-------------------|-------------------|-------------------|------------------|------------------|------------------|
| **SGA (top-k)**     | **BLEU ↑**     | **24.88**₀.₄₁    | **24.92**₀.₁₄    | **25.49**₀.₄₁    | **25.78**₀.₀₈   | 19.93₁.₉₂        | 23.43₁.₂₀        |
|                     | **METEOR ↑**   | **19.82**₀.₂₉    | **19.77**₀.₀₈    | **20.15**₀.₂₁    | 20.05₀.₀₇       | 16.80₀.₆₀        | 18.81₀.₈₂        |
|                     | **ROUGE-1 ↑**  | **30.27**₀.₂₂    | **30.44**₀.₄₉    | **30.55**₀.₂₅    | **30.67**₀.₀₉   | 28.02₀.₄₈        | 28.98₀.₈₆        |
| **SGA (top-1)**     | **BLEU ↑**     | 24.57₀.₁₇        | 24.55₀.₁₅        | 24.57₀.₂₂        | 25.72            | 19.93₁.₉₂        | 23.43₁.₂₀        |
|                     | **METEOR ↑**   | 19.47₀.₂₉        | 19.48₀.₀₇        | 19.52₀.₁₂        | **20.25**        | 16.80₀.₆₀        | 18.81₀.₈₂        |
|                     | **ROUGE-1 ↑**  | 29.06₀.₀₉        | 29.08₀.₀₉        | 29.28₀.₁₂        | 30.53            | 28.02₀.₄₈        | 28.98₀.₈₆        |
| **KSA**             | **BLEU ↑**     | 15.98₀.₅₆        | 16.53₀.₂₀        | 17.41₀.₂₆        | 24.89₀.₅₆       | 19.93₁.₉₂        | 23.43₁.₂₀        |
|                     | **METEOR ↑**   | 13.38₀.₄₃        | 13.35₀.₀₉        | 14.11₀.₂₂        | 18.80₀.₄₃       | 16.80₀.₆₀        | 18.81₀.₈₂        |
|                     | **ROUGE-1 ↑**  | 19.05₀.₄₄        | 19.25₀.₂₂        | 20.93₀.₁₅        | 29.16₀.₄₄       | 28.02₀.₄₈        | 28.98₀.₈₆        |
| **KSA w/o public**  | **BLEU ↑**     | 15.65₀.₄₅        | 16.36₀.₂₁        | 16.13₀.₅₂        | 24.03₀.₁₄       | 19.93₁.₉₂        | 23.43₁.₂₀        |
|                     | **METEOR ↑**   | 12.80₀.₃₆        | 13.35₀.₁₀        | 13.23₀.₄₁        | 18.15₀.₃₂       | 16.80₀.₆₀        | 18.81₀.₈₂        |
|                     | **ROUGE-1 ↑**  | 18.23₀.₃₉        | 19.61₀.₃₂        | 20.93₀.₁₅        | 27.38₀.₃₅       | 28.02₀.₄₈        | 28.98₀.₈₆        |

SGA (top-$k$) denotes our method $k$ candiates and SGA (top-$1$) denotes direct selection method skipping the stage.   
