"""
2018-10-01
(c) A. Martel Lab Co.
author: G.Kuling
This is my predictor code.
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
"""

import numpy as np
import pandas as pd
from keras.models import model_from_json
import os
from TrainingUtils import dice_loss, dice_coef, dice_coefficient_numpy_arrays
import DGenerator as generator

class Predictor:
    def __init__(self,
                 model_folder,
                 data_folder,
                 target_folder,
                 ofolder,
                 opt,
                 testing_direction=True):
        '''
        Initialization criteria
        :param model_folder: directory where weights and JSON file are saved
        of the model.
        :param data_folder: directory where testing data is stored. File
        Format: nii.gz
        :param target_folder: corresponding ground truth segmentations for
        the testing data. File Format: nii.gz
        :param ofolder: Output directory to save the numpy array and csv of
        DSC for each test subject
        :param opt: Optimizer used to trian the model, so it can be compiled.
        :param testing_direction: Direction that the generator is initialized to
        perform. True=input image, output segmentation, False=input
        segmentation, output image.
        '''

        json_file_name = [i for i in os.listdir(model_folder) if i.endswith('json')][0]
        weights_file_name = [i for i in os.listdir(model_folder) if i.startswith('model_best')][0]
        json_file = open(''.join([model_folder, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load the weights to the model
        self.model.load_weights(''.join([model_folder, '/', weights_file_name]))
        self.model.compile(loss=dice_loss, metrics=[dice_coef],
                           optimizer=opt)
        # print('Model is ready to predict.')

        gen = generator.DGenerator(data_dir=data_folder,
                                   target_dir=target_folder,
                                   batch_size=1,
                                   regular=testing_direction,
                                   shuffle=False,
                                   num_classes=1)

        gen.batch_size = gen.__len__()

        self.test_set, self.test_tar = gen.__getitem__(0)

        self.columns = ['Volume Number', 'Dice coefficient']
        self.evaluation_data_frame = pd.DataFrame(columns=self.columns)
        self.ofolder = ofolder

    def predict_and_evaluate(self):
        """
        The main driver function of the predictor. Predicts on the given file
        and evaluates the DSC to the target maps
        :return: The mean and standarad deviation of each mask.
        WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
        """
        self.y_pred = self.model.predict(self.test_set)

        eval = self.model.evaluate(self.test_set, self.test_tar, batch_size=1)

        print(eval)

        if os.path.exists(self.ofolder) is False:

            os.makedirs(self.ofolder)

        np.save(''.join([self.ofolder, '/pred_TestSet.npy']), self.y_pred)



        for index in range(self.y_pred.shape[0]):
            # prep the data
            predicted_volume = self.y_pred[index, :, :, 0].astype('float32')
            predicted_volume[predicted_volume >= 0.5] = 1
            predicted_volume[predicted_volume < 0.5] = 0
            target_volume = self.test_tar[index, :, :, 0]

            dice_coefficient = dice_coefficient_numpy_arrays(target_volume[:, :],
                                                             predicted_volume[:, :])


            error_data_dictionary = {self.columns[0]: index,
                                     self.columns[1]: dice_coefficient}

            self.evaluation_data_frame = self.evaluation_data_frame.append(
                error_data_dictionary,
                ignore_index=True)


        dice_coefs_BG = self.evaluation_data_frame[self.columns[1]].values
        print('Mean DSC= ' + str(np.mean(dice_coefs_BG)))
        print('Std DSC= ' + str(np.std(dice_coefs_BG)))

        self.evaluation_data_frame.to_csv(
            ''.join([self.ofolder, '/evaluation_results.csv']))

        return [np.mean(self.evaluation_data_frame[self.columns[1]]),
                np.std(self.evaluation_data_frame[self.columns[1]])]


if __name__ == "__main__":
    # Oberon
    data_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
               '/imagesTs/'
    target_dir = '/jaylabs/amartel_data2/prostate_data/Task05_Prostate' \
                 '/labelsTs/'
    model_folder = '/home/gkuling/2019-03-30-CSC2541Project/UNet_regular/'
    ofolder = '/home/gkuling/2019-03-30-CSC2541Project/UNet_regular' \
              '/test_results/'

    a = Predictor(model_folder=model_folder,
                  data_folder=data_dir,
                  target_folder=target_dir,
                  ofolder=ofolder,
                  opt='ADAM',
                  testing_direction=True)
    a.predict_and_evaluate()
    print('done')
