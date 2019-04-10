import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import SimpleITK as sitk
import pandas as pd
import os

import DGenerator as generator
from TrainingUtils import dice_loss, dice_coef, dice_coefficient_numpy_arrays
from keras.models import model_from_json


if __name__ == '__main__':

    data_dir = 'Y:/prostate_data/Task05_Prostate/imagesTs/'

    target_dir = 'Y:/prostate_data/Task05_Prostate/labelsTs/'

    # model_folder = 'Z:/2019-03-30-CSC2541Project/UNet_regular_grey2/'
    model_folder = r'Y:\GeoffKlein\CSC2541_Project\ModelOutputs' \
                   r'\UNet_reuglarWAugcGAN_rev2\1500_syn_samples\\'


    json_file_name = [i for i in os.listdir(model_folder) if i.endswith('json')][0]
    weights_file_name = [i for i in os.listdir(model_folder) if i.startswith('model_best')][0]
    json_file = open(''.join([model_folder, '/', json_file_name]))
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Load the weights to the model
    model.load_weights(''.join([model_folder, '/', weights_file_name]))
    model.compile(loss=dice_loss, metrics=[dice_coef],
                  optimizer='ADAM')

    gen = generator.DGenerator(data_dir=data_dir,
                               target_dir=target_dir,
                               batch_size=1,
                               regular=True,
                               shuffle=False)

    gen.batch_size = gen.__len__()

    test_set, test_tar = gen.__getitem__(0)

    y_pred_hold = model.predict(test_set, verbose=1)

    # y_pred[y_pred >= 0.5] = 1
    # y_pred[y_pred < 0.5] = 0

    y_pred = np.where(y_pred_hold >= 0.5, 1, 0)

    dice_coefficient = []

    for index in range(y_pred.shape[0]):
        # prep the data
        predicted_volume = y_pred[index, :, :, :].astype('float32')
        target_volume = test_tar[index, :, :, :]

        dice_coefficient_hold = dice_coefficient_numpy_arrays(target_volume[
                                                              :, :,0],
                                                              predicted_volume[:, :,0])

        dice_coefficient.append(dice_coefficient_hold)


    dice_coefficient = np.array(dice_coefficient)


    sort_idx = np.argsort(dice_coefficient)[::-1]

    hold = [1, 18]

    idx_hold0 = sort_idx[0 + hold[0]]

    idx_hold1 = sort_idx[1 + hold[1]]

    # idx_hold_tot = [idx_hold0, idx_hold1]

    # idx_hold_tot = [238, 211]

    idx_hold_tot = [211, 212]

    fig = plt.figure(figsize=(6, 6))
    sub_plot_idx = 0

    for idx in range(2):

        # idx_hold = sort_idx[idx + hold[idx]]

        idx_hold = idx_hold_tot[idx]

        DSC = dice_coefficient[idx_hold]

        # print(idx_hold)

        ground_truth_data = test_tar[idx_hold, :, :,0]
        pred_data = y_pred[idx_hold, :, :,0]
        test_data = test_set[idx_hold, :, :,0]


        ax1 = fig.add_subplot(2, 2, sub_plot_idx + 1)
        sub_plot_idx += 1
        plt1 = ax1.imshow(test_data, cmap='gray', label='Ground Truth Segmentation')

        ax1.set_title('MRI Slice of Prostate', fontsize=14, fontname='Times New Roman')


        ax2 = fig.add_subplot(2, 2, sub_plot_idx + 1)
        sub_plot_idx += 1

        plt1 = ax2.imshow(ground_truth_data, cmap='Greys', label='Ground Truth Segmentation')
        plt2 = ax2.imshow(test_data, cmap='gray', alpha=0.5, label='Testing Sample')
        plt3 = ax2.contour(pred_data, colors='r', linewidths=0.5, label='Segmentation Prediction')

        ax2.set_title(r'DSC = {:.2f}'.format(DSC), fontsize=14, fontname='Times New Roman')

        ax1.axis('off')
        ax2.axis('off')


        # labels = ['a', 'b', 'c', 'd', 'e', 'f']
        # ax.text(0.1, 0.9, labels[idx],
        #         horizontalalignment='center',
        #         verticalalignment='center',
        #          transform=ax.transAxes,
        #          fontsize=14)


        # plt.subplots_adjust(hspace=.5, wspace=.001)



    red_patch = mpatches.Patch(color='red', label='Contour of Predicted Segmentation')
    black = mpatches.Patch(color='k', label='Ground Truth (GT) Segmentation')
    plt.figlegend(handles=[red_patch, black], loc='lower center', fontsize=12)

    # ax.legend()
    # plt.suptitle('Axial Vertebral Bodies and Associated Segmentations', fontsize=14, fontname='Times New Roman')
    plt.show()
    print('done')
    # plt.savefig('segmentation_best_regular.png', bbox_inches='tight', dpi=650)
    #
    # plt.savefig('segmentation_best.svg', box_inches='tight', dpi=650)
