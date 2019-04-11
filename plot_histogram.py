import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':



    csv_dir = r'Y:\GeoffKlein\CSC2541_Project\ModelOutputs\UNet_regular_rev2\test_results\evaluation_results.csv'

    csv_dir = 'ModelOutputs/UNet_regularWAugcGAN_rev2/1500_syn_samples/test_results/evaluation_results.csv'


    df = pd.read_csv(csv_dir, index_col=False)

    dsc_list = df['Dice coefficient'].values

    bin_range = np.linspace(0, 1, 31)


    fig = plt.figure(figsize=(4.5, 7))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim([0, 70])
    plt.hist(dsc_list, bins=bin_range)
    plt.title('DSC of Validation Samples For the UNet baseline')
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


    # plt.savefig('test_dsc_freq_regular_unet.png', bbox_inches='tight', dpi=650)
    #
    # plt.savefig('test_dsc_freq.svg', bbox_inches='tight', dpi=650)