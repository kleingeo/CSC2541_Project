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
from Utils import My_new_loss
from Evaluation_Measures import dice_coefficient_numpy_arrays


class Predictor:
    def __init__(self,
                 model_folder,
                 batch_file,
                 target_file,
                 ofolder,
                 opt):
        """
        Initializer for the Predictor code
        :param model_folder: The folder that has the json and weights file saved
        :param batch_file: The file you wish to evaluate
        :param target_file: The target of the file you wish to evaluate
        :param ofolder: Where everything is saved at the end
        :param opt: The optimizer used.
        """
        json_file_name = \
            [i for i in os.listdir(model_folder) if i.endswith('json')][0]
        weights_file_name = \
            [i for i in os.listdir(model_folder) if i.startswith(
                'model_best')][0]
        json_file = open(''.join([model_folder, '/', json_file_name]))
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load the weights to the model
        self.model.load_weights(''.join([model_folder, '/', weights_file_name]))
        self.model.compile(loss=My_new_loss, metrics=[My_new_loss],
                           optimizer=opt)
        print('Model is ready to predict.')

        self.test_set = np.load(batch_file)
        self.test_tar = np.load(target_file)

        self.columns = ['Volume Number', 'Dice coefficient-BG',
                        'Dice coefficient-FT', 'Dice coefficient-FGT']
        self.evaluation_data_frame = pd.DataFrame(columns=self.columns)
        self.ofolder = ofolder

    def predict_and_evaluate(self):
        """
        The main driver function of the predictor. Predicts on the given file
        and evaluates the DSC to the target maps
        :return: The mean and standard deviation of each mask.
        WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
        """
        self.y_pred = self.model.predict(self.test_set,
                                         verbose=1)

        np.save(''.join([self.ofolder, '/pred_TestSet.npy']), self.y_pred)

        for index in range(self.y_pred.shape[0]):
            # prep the data
            predicted_volume = self.y_pred[index, :, :, :].astype('float32')
            predicted_volume[predicted_volume >= 0.5] = 1
            predicted_volume[predicted_volume < 0.5] = 0
            predicted_volume = np.squeeze(predicted_volume)
            target_volume = self.test_tar[index, :, :, :]
            dice_coefficient_BG = \
                dice_coefficient_numpy_arrays(target_volume[0, :, :],
                                              predicted_volume[0, :, :])
            dice_coefficient_FT = \
                dice_coefficient_numpy_arrays(target_volume[1, :, :],
                                              predicted_volume[1, :, :])
            dice_coefficient_FGT = \
                dice_coefficient_numpy_arrays(target_volume[2, :, :],
                                              predicted_volume[2, :, :])

            error_data_dictionary = \
                {self.columns[0]: index,
                 self.columns[1]: dice_coefficient_BG,
                 self.columns[2]: dice_coefficient_FT,
                 self.columns[3]: dice_coefficient_FGT}
            self.evaluation_data_frame = self.evaluation_data_frame.append(
                error_data_dictionary,
                ignore_index=True)
            print(error_data_dictionary)

        dice_coefs_BG = self.evaluation_data_frame[self.columns[1]]
        print('Mean DSC BG= ' + str(np.mean(dice_coefs_BG)))
        print('Std DSC BG= ' + str(np.std(dice_coefs_BG)))

        dice_coefs_BG = self.evaluation_data_frame[self.columns[2]]
        print('Mean DSC FT= ' + str(np.mean(dice_coefs_BG)))
        print('Std DSC FT= ' + str(np.std(dice_coefs_BG)))

        dice_coefs_BG = self.evaluation_data_frame[self.columns[3]]
        print('Mean DSC FGT= ' + str(np.mean(dice_coefs_BG)))
        print('Std DSC FGT= ' + str(np.std(dice_coefs_BG)))

        self.evaluation_data_frame.to_csv(
            ''.join([self.ofolder, '/evaluation_results.csv']))

        return [np.mean(self.evaluation_data_frame[self.columns[1]]),
                np.std(self.evaluation_data_frame[self.columns[1]]),
                np.mean(self.evaluation_data_frame[self.columns[2]]),
                np.std(self.evaluation_data_frame[self.columns[2]]),
                np.mean(self.evaluation_data_frame[self.columns[3]]),
                np.std(self.evaluation_data_frame[self.columns[3]])]


if __name__ == "__main__":
    ### Changeable Variables
    # For Oberon
    # model_folder = r'/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #                r'/Models/RedoFolder/'
    # batch_file = r'/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data/Batches' \
    #              r'/2Ch/ParamValid' \
    #              r'/data0.npy'
    # target_file = r'/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data' \
    #               r'/Batches/2Ch/ParamValid' \
    #               r'/target0.npy'
    # ofolder = r'/jaylabs/amartel_data2/Grey/2018-06-11-FGTSeg-Data/Models' \
    #           r'/RedoFolder/'

    # For Titania
    model_folder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                   '/Models/HPGS1-UNet/[ADAM]/[\'t_dilRate\', \'t_depth\', ' \
                   '\'t_dropOut\']/[2.0, 5.0, 0.25, ]/'
    batch_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                 r'/Batches/2D/Testdata0.npy'
    target_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                 r'/Batches/2D/Testtarget0.npy'
    ofolder = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
              r'/TestResults/UNet[ADAM, 2, 5, 0.25]/'

    a = Predictor(model_folder=model_folder,
                  batch_file=batch_file,
                  target_file=target_file,
                  ofolder=ofolder,
                  opt='ADAM')
    a.predict_and_evaluate()
    print('done')
