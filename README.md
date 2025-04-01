# BondValenceParametersFit
A Python package for fitting bond valence parameters $R_0$ and $B$ for cation-anion pairs using Materials Project data. It includes two modules: (1) computing theoretical bond valence and (2) optimizing parameters by matching computed and empirical values. This tool refines bond valence analysis with a data-driven approach.

## How to fit BV parameters

```
from bond_valence_processor import BondValenceProcessor

cations = ['Li'] # a list of cation species 
my_api_key = "your_api_key"
algos = ['shgo', 'brute', 'diff', 'dual_annealing', 'direct']
processor = BondValenceProcessor(my_api_key, algos, cations)
    
for cation in cations:
    processor.process_cation_system(cation)
    
```

