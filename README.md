# Advanced Topics in Security (COMS4507) Project - IoT NIDS
#### Author: Tompnyx
##### Student Number: 45872093
## Introduction
This project is both an anomaly detection and traffic classification application to the UQ Machine Learning-Based NIDS Datasets
### How to Run
To run the program, main.py should be run. However, two conditions must be met: the dataset directory and the output directory. Additional arguments antecedent with '**--**' include:
- **ext** [String] (Specify the file extension format. The default is '.csv'.)
- **classification** or **anomaly** [Boolean] (Whether the classification or anomaly approaches should be used. Defaults to **classification**.)
- **show_heatmap** [Boolean] (Shows a heatmap of all the parameters.)
- **show_comparisons** [Boolean] (Shows a comparison between the used algorithms.)
- **test_split** [Float] (Sets the test to train split for the dataset with the stated percentage related to the test subset size.)
- **seed** [Integer] (Sets the seed used to control the shuffling applied and allows for reproducible results)

For example:
```commandline
python3 .\main.py .\COMS4507_NIDS\NetFlow-v1-Datasets\NF-CSE-CIC-IDS2018.csv .\COMS4507\COMS4507_NIDS\Output --anomaly --show_heatmap --show_comparisons
```
### Dataset Description
The dataset used for training and testing was the NF-CSE-CIC-IDS2018 dataset, which is a generated NetFlow-based dataset. The dataset has 8,392,401 flows, of which 1,019.203 or (12.4%) are attack samples. These attack samples include:

- BruteForce
- Bot
- DoS
- DDoS
- Infiltration
- Web Attacks

However, any of the datasets from the [Machine Learning-Based NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA4) page can be used.

### Algorithms Used
Six different algorithms are utilised from the scikit-learn package:

- Random Forest (RF) [Classification]
- k-nearest neighbour (kNN) [Classification]
- Support Vector Classifier (SVC) [Classification]
- Gaussian Naive Bayes (GNB) [Anomaly]
- Decision Tree (DT) [Anomaly]
- Logistic Regression (LR) [Anomaly]

### Dependencies
- Python (Tested: 3.9.7)
- matplotlib (Tested: 3.5.1)
- numpy (Tested: 1.21.4)
- pandas (Tested: 1.3.5)
- seaborn (Tested: 0.11.2)
- scikit-learn (Tested: 1.0.1)