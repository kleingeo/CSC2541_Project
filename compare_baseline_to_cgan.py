import numpy as np
from scipy import stats
import pandas as pd
import os

if __name__ == '__main__':

    syn_results_csv = 'ModelOutputs/UNet_regularWAugcGAN_rev2/1700_syn_samples/test_results/evaluation_results.csv'

    baseline_results_csv = 'ModelOutputs/UNet_regular_rev2/test_results/evaluation_results.csv'

    dsc_baseline = pd.read_csv(baseline_results_csv)['Dice coefficient'].values

    dsc_syn = pd.read_csv(syn_results_csv)['Dice coefficient'].values

    print(stats.mannwhitneyu(dsc_baseline, dsc_syn, alternative='less'))

    print()
