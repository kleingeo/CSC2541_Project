import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TrainingUtils import dice_loss, dice_coef
import DGenerator as generator
from keras.models import model_from_json
import os



if __name__ == '__main__':
    model_folder_pretrain = 'ModelOutputs/UNetAugmentor_rev3/'
    model_folder_cgan = 'ModelOutputs/cGANUnetAugmentor_rev2/'

    data_dir_train = 'D:/prostate_data/Task05_Prostate/imagesTr/'
    target_dir_train = 'D:/prostate_data/Task05_Prostate/labelsTr/'

    data_dir_val = 'D:/prostate_data/Task05_Prostate/imagesTs/'
    target_dir_val = 'D:/prostate_data/Task05_Prostate/labelsTs/'

    # Load pretrain model
    json_file_name = [i for i in os.listdir(model_folder_pretrain) if i.endswith('json')][0]
    weights_file_name = [i for i in os.listdir(model_folder_pretrain) if i.startswith('model_best')][0]
    json_file = open(''.join([model_folder_pretrain, '/', json_file_name]))
    loaded_model_json = json_file.read()
    json_file.close()
    model_pretrain = model_from_json(loaded_model_json)

    model_pretrain.load_weights(''.join([model_folder_pretrain, '/', weights_file_name]))
    model_pretrain.compile(loss=dice_loss, metrics=[dice_coef], optimizer='ADAM')

    # Load cGAN model
    json_file_name = [i for i in os.listdir(model_folder_cgan) if i.endswith('json')][0]
    weights_file_name = [i for i in os.listdir(model_folder_cgan) if i.startswith('model_best')][0]
    json_file = open(''.join([model_folder_cgan, '/', json_file_name]))
    loaded_model_json = json_file.read()
    json_file.close()
    model_cgan = model_from_json(loaded_model_json)

    model_cgan.load_weights(''.join([model_folder_cgan, '/', weights_file_name]))
    model_cgan.compile(loss=dice_loss, metrics=[dice_coef], optimizer='ADAM')

    gen = generator.DGenerator(data_dir=data_dir_val,
                               target_dir=target_dir_val,
                               batch_size=1,
                               regular=True,
                               shuffle=False,
                               num_classes=1)
    test_set, test_tar = gen.__getitem__(5)

    pred_img_pretrain = model_pretrain.predict(test_tar)
    pred_img_cgan = model_cgan.predict(test_tar)

    fig = plt.figure(figsize=(15, 7))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(test_tar[0, :, :, 0])
    ax1.set_title('Original Segmentation')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(pred_img_cgan[0, :, :, 0])
    ax2.set_title('Synthetic Image')
    ax2.axis('off')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(test_set[0, :, :, 0])
    ax3.set_title('Original Image')
    ax3.axis('off')

    plt.show()

    plt.savefig('../cGAN_val.png', bbox_inches='tight', dpi=650)