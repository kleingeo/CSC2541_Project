import numpy as np
import pandas as pd
import os

if __name__ == "__main__":


    main_model_folder = 'ModelOutputs/UNet_regularWAugcGAN_rev2'

    dsc_avg = []
    dsc_std = []

    syn_amount = []

    for syn_sample_folder in os.listdir(main_model_folder):

        csv_folder_path = '/'.join([main_model_folder,
                                    syn_sample_folder,
                                    'test_results/evaluation_results.csv'])

        syn_sample_number = int(syn_sample_folder.split('_')[0])

        df = pd.read_csv(csv_folder_path)

        dsc_avg_hold = df['Dice coefficient'].values.mean()
        dsc_std_hold = df['Dice coefficient'].values.std()

        dsc_avg.append(dsc_avg_hold)
        dsc_std.append(dsc_std_hold)
        syn_amount.append(syn_sample_number)

    dsc_avg = np.array(dsc_avg)
    dsc_std = np.array(dsc_std)

    syn_amount = np.array(syn_amount)

    sort_idx = np.argsort(syn_amount)

    syn_amount = syn_amount[sort_idx]
    dsc_avg = dsc_avg[sort_idx]
    dsc_std = dsc_std[sort_idx]

    max_syn_amount_idx = dsc_avg.argmax()

    print('{} synthetic samples have the highest average DSC.'.format(syn_amount[max_syn_amount_idx]))

    print(u'Highest average DSC \u00b1 standard deviation were: {:0.2f} \u00b1 {:0.2f}'.format(
        dsc_avg[max_syn_amount_idx],
        dsc_std[max_syn_amount_idx]))


    baseline_results_csv = 'ModelOutputs/UNet_regular_rev2/test_results/evaluation_results.csv'

    dsc_baseline = pd.read_csv(baseline_results_csv)['Dice coefficient'].values

    dsc_baseline_avg = dsc_baseline.mean()
    dsc_baseline_std = dsc_baseline.std()

    print(u'The baseline U-Net had an average DSC \u00b1 standard deviation of: {:0.2f} \u00b1 {:0.2f}'.format(
        dsc_baseline_avg,
        dsc_baseline_std))

