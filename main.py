import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# 1. Take user input
parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str, help='csv file containing the NetFlow-based dataset')
parser.add_argument('out_dir', type=str, help='output directory')
parser.add_argument('--ext', type=str, default='.csv', help='the filename extension (default: .csv)')
args = parser.parse_args()

# Generate the output directory if it doesn't exist
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
    print('Output directory automatically generated')

# 2. Load the dataset
nf_dataset = pd.read_csv(args.dataset_dir, header=0)
print(nf_dataset.head())
