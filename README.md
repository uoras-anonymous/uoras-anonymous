## README
This is an anonymous repository for NeurIPS submission. 

To train the non-stationary version of UORAS
`python train_N_subdomains_fixed_train_tol_valid.py`

To train the stationary version of UORAS
`python train_N_subdomains_stationary.py`

To test the non-statioary trained UORAS model use
`python test_N_subdomains_tolerance.py`

To test the stationary UORAS as well as analytical models (T2, T2B, OO2) use
`python test_N_subdomains_tolerance_stationary.py`

To test the stationary UORAS as well as analytical models (T2, T2B, OO2) as a GMRES preconditioner use
`python test_N_subdomains_tolerance_stationary_gmres.py`

The dataset size exceeds the size limits of GitHub. However, you can easily generate it using 
`python datagen.py`

If you have any questions or problems running the code, please state them during the rebuttal and I will answer them.