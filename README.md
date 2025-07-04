# Repository for "Reconciling Privacy and Explainability in High-Stakes: A Systematic Inquiry" (Accepted at TMLR)

This repository accompanies the paper **"Reconciling Privacy and Explainability in High-Stakes: A Systematic Inquiry"**, accepted at **Transactions on Machine Learning Research (TMLR)**. 

## Overview of Scripts

1. **`vanilla-training.py`**  
   Trains models without any privacy preservation (non-private training). 
2. **`dp-training-with-opacus.py`**  
   Trains models with differential privacy using the **Opacus** library.
3. **`LA-and-PIS.py`**  
   Performs analysis related to the **Localization Assumption (LA)** and calculates **PIS**, as described in Section 4 of the paper.
4. **`dcka-analysis.py`**  
   Conducts **DCKA** analysis as outlined in Section 7 of the paper.
5. **`representation-and-sensitivity-hypothesis.py`**  
   Conducts statistical significance tests for *representations* and *sensitivity* on a layer-by-layer basis, as detailed in Section 7 of the paper.


