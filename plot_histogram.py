import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    csv_dir_syn = 'ModelOutputs/UNet_regularWAugcGAN_rev2/1700_syn_samples/test_results/evaluation_results.csv'
    csv_dir_baseline = 'ModelOutputs/UNet_regular_rev2/test_results/evaluation_results.csv'

    df_base = pd.read_csv(csv_dir_baseline, index_col=False)
    dsc_base_list = df_base['Dice coefficient'].values

    df_syn = pd.read_csv(csv_dir_syn, index_col=False)
    dsc_syn_list = df_syn['Dice coefficient'].values

    bin_range = np.linspace(0, 1, 31)

    fig = plt.figure(figsize=(15, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_ylim([0, 70])
    ax1.hist(dsc_base_list, bins=bin_range)
    ax1.set_title('DSC of Validation Samples For the U-Net baseline')
    plt.xlabel('Binned DSC')
    plt.ylabel('Number of Samples')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_ylim([0, 70])
    ax2.hist(dsc_syn_list, bins=bin_range)
    ax2.set_title('DSC of Validation Samples For the U-Net synthetic')
    plt.xlabel('Binned DSC')
    plt.ylabel('Number of Samples')

    # ax.text(0.5, -0.1, 'b',
    #         horizontalalignment='center',
    #         verticalalignment='center',
    #         transform=ax.transAxes,
    #         fontsize=14)


    plt.show()

    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    print('done')


    plt.savefig('../test_dsc_freq_regular_unet.png', bbox_inches='tight', dpi=650)
    #
    # plt.savefig('test_dsc_freq.svg', bbox_inches='tight', dpi=650)