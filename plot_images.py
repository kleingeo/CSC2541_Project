import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

import DGenerator as generator
from TrainingUtils import dice_loss, dice_coef, dice_coefficient_numpy_arrays
from keras.models import model_from_json


if __name__ == '__main__':

    data_dir = 'D:/prostate_data/Task05_Prostate/imagesTs/'
    target_dir = 'D:/prostate_data/Task05_Prostate/labelsTs/'
    model_folder = 'ModelOutputs/UNet_regularWAugcGAN_rev2/1500_syn_samples/'
    #
    # model_folder = 'ModelOutputs/UNet_regular_rev2/'



    # json_file_name = [i for i in os.listdir(model_folder) if i.endswith('json')][0]
    # weights_file_name = [i for i in os.listdir(model_folder) if i.startswith('model_best')][0]
    # json_file = open(''.join([model_folder, '/', json_file_name]))
    # loaded_model_json = json_file.read()
    # json_file.close()
    # model = model_from_json(loaded_model_json)
    #
    # # Load the weights to the model
    # model.load_weights(''.join([model_folder, '/', weights_file_name]))
    # model.compile(loss=dice_loss, metrics=[dice_coef],
    #               optimizer='ADAM')

    gen = generator.DGenerator(data_dir=data_dir,
                               target_dir=target_dir,
                               batch_size=1,
                               regular=True,
                               shuffle=False)

    gen.batch_size = gen.__len__()

    test_set, test_tar = gen.__getitem__(0)

    # y_pred_hold = model.predict(test_set, verbose=1)
    y_pred_hold = np.load(model_folder + '/test_results/pred_TestSet.npy')

    y_pred = np.where(y_pred_hold >= 0.5, 1, 0)

    pred_shape = y_pred.shape

    # If prediction is accidentally channels first, fix
    if pred_shape[1] != pred_shape[2]:

        y_pred = np.rollaxis(y_pred, axis=1, start=4)


    dice_coefficient = []

    for index in range(y_pred.shape[0]):
        # prep the data
        predicted_volume = y_pred[index, :, :, :].astype('float32')
        target_volume = test_tar[index, :, :, :]

        dice_coefficient_hold = dice_coefficient_numpy_arrays(target_volume[:, :, 0],
                                                              predicted_volume[:, :, 0])

        dice_coefficient.append(dice_coefficient_hold)


    dice_coefficient = np.array(dice_coefficient)

    sort_idx = np.argsort(dice_coefficient)[::-1]


    idx_hold_tot = [211, 212]

    fig = plt.figure(figsize=(6, 6))
    sub_plot_idx = 0

    for idx in range(2):

        idx_hold = idx_hold_tot[idx]

        DSC = dice_coefficient[idx_hold]

        ground_truth_data = test_tar[idx_hold, :, :, 0]
        pred_data = y_pred[idx_hold, :, :, 0]
        test_data = test_set[idx_hold, :, :, 0]


        ax1 = fig.add_subplot(2, 2, sub_plot_idx + 1)
        sub_plot_idx += 1
        plt1 = ax1.imshow(test_data, cmap='gray')

        ax1.set_title('MRI Slice of Prostate', fontsize=14, fontname='Times New Roman')


        ax2 = fig.add_subplot(2, 2, sub_plot_idx + 1)
        sub_plot_idx += 1

        plt1 = ax2.imshow(ground_truth_data, cmap='Greys')
        plt2 = ax2.imshow(test_data, cmap='gray', alpha=0.5)
        plt3 = ax2.contour(pred_data, colors='r', linewidths=0.5)

        ax2.set_title(r'DSC = {:.2f}'.format(DSC), fontsize=14, fontname='Times New Roman')

        ax1.axis('off')
        ax2.axis('off')



    red_patch = mpatches.Patch(color='red', label='Contour of Predicted Segmentation')
    black = mpatches.Patch(color='k', label='Ground Truth (GT) Segmentation')
    plt.figlegend(handles=[red_patch, black], loc='lower center', fontsize=12)


    plt.show()
    # plt.savefig('segmentation_best_regular.png', bbox_inches='tight', dpi=650)
    #
    # plt.savefig('segmentation_best.svg', box_inches='tight', dpi=650)
