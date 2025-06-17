# PrecisionMatrixAdj

This repository contains the code for the paper **"Incorporating Interactive User Feedback through Precision Matrix Adjustments for High-Dimensional Anomaly Detection"**, currently in review at ICDM 2025.

## To Collect the Results from the Paper

1. Run example experiments:
    First, change the ```data_path``` variable in ```run_pma.sh``` to the path containing the processed data - typically this is ```./data/``` To collect all numbers from the paper results section for PMA and PMA-MGE, as well as the default Mahalanobis distance MD, simply run
    ```
    bash src/run_pma.sh
    ```

    These are automatically saved to the respective results folders.

2. Competitor experiments:
    GAOD uses the same code-base here. To collect these numbers, run 
    ```
    bash src/run_gaod.sh
    ```

    AAD-IF and AAD-LODA are found in [this repository](https://github.com/shubhomoydas/ad_examples/tree/master). F1 and AUPRC were calculated from anomaly scores and saved each iteration after the budget is exhausted. 

    FIF is found in [this repository](https://github.com/siddiqmd/FeedbackIsolationForest). Similarly, F1 and AUPRC were calculated each iteration using the anomaly scores. 

    SOEL is found in [this repository](https://github.com/aodongli/Active-SOEL/tree/main/NTL). Once again, the anomaly scores are used for F1 and AUPRC. This base was used for image embeddings

    Runtime calculations were done for the iterative portions of the algorithms, and any optimization or model updates. 

    Please see the paper for experimental setups in all cases (hyperparameters, model specifications, etc.). 

    Plots of runtime vs AUPRC/FI are constructed using ```python src/plotting.py```, and SOEL results are processed using ```python src/dl_comparison.py```. Statistical significance testing is done with ```python src/wilcoxon_tests.py```. Latex table creation is done with ```python src/process_fif.py```. In all cases, there are variables for the base directories at the start of each function that should be set. These should contain the respective results txt or csv files. Naming conventions on most files are fixed currently with the anomaly score processing for each algorithm. If collecting results from AAD-IF/LODA, FIF, or SOEL, these should be consistent, or modified accordingly. We include the collected and saved runtimes and F1/AUPRC scores from these four algorithms in their various results folders here for easy access, which were used for the paper. 

3. MSL and MNIST/KMNIST Pretrained Models:
    The pretrained models for the Mars Science Laboratory (MSL) and MNIST are found in ```./models/```. These were used to extract feature embeddings for these two datasets. 

4. Data:
    The processed data can be found at [this link](https://drive.google.com/file/d/1_aM7PEILJL1iriV8bHrO9A40cHhrDLRn/view?usp=drive_link). Vision datasets were processed using ```torch_processing.py```, outside of the SOEL experiments, which were saved from the SOEL repository code. 




