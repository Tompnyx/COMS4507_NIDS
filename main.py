#!/usr/bin/env python
"""Applies both anomaly detection and traffic classification to the UQ Machine Learning-Based NIDS Datasets

Running environment included packages:
matplotlib = 3.5.1
numpy = 1.21.4
pandas = 1.3.5
python = 3.9.7
seaborn = 0.11.2
scikit-learn - 1.0.1
"""

__author__ = "tompnyx"
__version__ = "1.0.0"

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import time

# 1. Take user input
parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str, help='csv file containing the NetFlow-based dataset')
parser.add_argument('out_dir', type=str, help='Output directory')
parser.add_argument('--ext', type=str, default='.csv', help='The filename extension (default: .csv)')
parser.add_argument('--supervised', action=argparse.BooleanOptionalAction, default=True, help='Perform traffic '
                                                                                              'classification ('
                                                                                              'supervised)')
parser.add_argument('--unsupervised', dest='supervised', action='store_false', help='Perform anomaly detection ('
                                                                                    'un-supervised)')
parser.add_argument('--show_heatmap', action=argparse.BooleanOptionalAction,
                    help='Whether to calculate and show a heatmap of the features')
parser.add_argument('--show_comparisons', action=argparse.BooleanOptionalAction,
                    help='Whether to calculate and show comparisons between the different algorithms')
parser.add_argument('--test_split', type=float, default=0.33, help='The default test to train split for the dataset '
                                                                   'with the stated percentage related to the test '
                                                                   'subset size')
parser.add_argument('--seed', type=int, default=42, help='The seed used to control the shuffling applied and allows '
                                                         'for reproducible results')
args = parser.parse_args()

# Generate the output directory if it doesn't exist
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
    print('Output directory automatically generated')

# 2. Load the dataset
nf_dataset = pd.read_csv(args.dataset_dir, header=0)
print('Sample of the inputted dataset: \n', nf_dataset.head())
print('\nShape of the dataset: ', nf_dataset.shape)

# Create the feature and label subsets
nf_features = nf_dataset.copy()
nf_labels_supervised = nf_features.pop('Attack')
nf_labels_unsupervised = nf_features.pop('Label')
print('List of null values in each feature: \n', nf_features.isnull().sum())

# Plot a heatmap of feature correlations
if args.show_heatmap:
    plt.figure(figsize=(15, 12))
    sns.heatmap(nf_features.corr())
    corr = nf_features.corr()
    plt.show()

# Convert IP Addresses to numerical labels
le_src = preprocessing.LabelEncoder()
le_src.fit(nf_features['IPV4_SRC_ADDR'])
nf_features['IPV4_SRC_ADDR'] = le_src.transform(nf_features['IPV4_SRC_ADDR'])

le_dst = preprocessing.LabelEncoder()
le_dst.fit(nf_features['IPV4_DST_ADDR'])
nf_features['IPV4_DST_ADDR'] = le_dst.transform(nf_features['IPV4_DST_ADDR'])

# # Convert to numpy array
# nf_features = np.array(nf_features)
# nf_labels_supervised = np.array(nf_labels_supervised)
# nf_labels_unsupervised = np.array(nf_labels_unsupervised)
# print("Shape of the features: ", nf_features.shape)

# Normalise the features
sc = preprocessing.MinMaxScaler()
nf_features = sc.fit_transform(nf_features)

if args.supervised:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(nf_features,
                                                                        nf_labels_supervised,
                                                                        test_size=args.test_split,
                                                                        random_state=args.seed)
else:
    x_train, x_test, y_train, y_test = model_selection.train_test_split(nf_features,
                                                                        nf_labels_unsupervised,
                                                                        test_size=args.test_split,
                                                                        random_state=args.seed)
print('\nShape of the training dataset: ', x_train.shape, y_train.shape)
print('Shape of the testing dataset: ', x_test.shape, y_test.shape)

# 3. Initialise and create the machine learning techniques

# Sections 3 and 4 were inspired from the related Geeks for Geeks webpage.
# "Intrusion Detection System Using Machine Learning Algorithms - GeeksforGeeks", GeeksforGeeks, 2022. [Online].
# Available: https://www.geeksforgeeks.org/intrusion-detection-system-using-machine-learning-algorithms/.
# [Accessed: 05- May- 2022].


def run_model(model):
    start_time = time.time()
    model.fit(x_train, y_train.values.ravel())
    test_end_time = time.time() - start_time
    print("Training time: ", test_end_time)
    start_time = time.time()
    y_test_pred = model.predict(x_train)
    train_end_time = time.time() - start_time
    print("Testing time:  ", train_end_time)
    print("Train score:   ", model.score(x_train, y_train))
    print("Test score:    ", model.score(x_test, y_test))
    return y_test_pred, test_end_time, model.score(x_train, y_train), train_end_time, model.score(x_test, y_test)


if args.supervised:
    # Traffic classification methods
    print('\nClassifying with a Random Forest implementation')
    rf = run_model(RandomForestClassifier(max_depth=4, verbose=1, n_jobs=16))
    print('\nClassifying with a k-nearest neighbour implementation')
    knn = run_model(KNeighborsClassifier(n_jobs=16))
    print('\nClassifying with a Support Vector Classifier implementation')
    svc = run_model(SVC(kernel='poly', verbose=1))
    algorithms = ['RF', 'kNN', 'SVC']
else:
    # Anomaly detection methods
    print('\nClassifying with a Gaussian Naive Bayes implementation')
    gnb = run_model(GaussianNB())
    print('\nClassifying with a Decision Tree implementation')
    dt = run_model(DecisionTreeClassifier(criterion="entropy"))
    print('\nClassifying with a Logistic Regression implementation')
    lr = run_model(LogisticRegression(max_iter=1200, verbose=1, n_jobs=16))
    algorithms = ['GNB', 'DT', 'LR']

# 4. Evaluate the performances of each algorithm


def compare_models(metric, value_index):
    if args.supervised:
        values = [rf[value_index], knn[value_index], svc[value_index]]
    else:
        values = [gnb[value_index], dt[value_index], lr[value_index]]
    title = 'Comparison of ' + metric

    fig, ax = plt.subplots()
    plt.title(title, size=15)
    bars = ax.barh(algorithms, values)
    ax.bar_label(bars)
    plt.show()


if args.show_comparisons:
    compare_models('Training Accuracy Score', 2)
    compare_models('Time taken to Train', 1)
    compare_models('Testing Accuracy Score', 4)
    compare_models('Time taken to Test', 3)
