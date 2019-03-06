"""
2019-03-02
author: Phil Boyer
Bare-bones implementation of segmentation pipeline to check functionality
"""

import numpy as np
import os
import Trainer
import Predictor as Predictor
import sys
import pandas as pd
import keras as k

np.random.seed(1)

class Run_Basic_Test:
    def __init__(self,
                 ofolder,
                 batch_folder,
                 target_folder,
                 valid_file,
                 target_file,
                 opt,
                 samples_per_card,
                 epochs,
                 gpus_used,
                 mode,
                 model_type):
        """
        Initializer for HP Grid Search of Numerical Values
        :param x_names: (list of strs) a list of variable names that will be
            grid searched over.
        :param D: (numpy array with 4 columns and len(x_names) rows of
        variables to grid search) This numpy array holds the parameters used
        in the grid search. column[0] min value of x[i], column[1] max value
        of x[i], column[2] is dx of x[i], column[3] is N stepd of x[i]
        :param ofolder: (str) string giving the directory you want to save
        all results in.
        :param batch_folder: (dir str) the location of data to be used by the
        data generator.
        :param target_folder: (dir str) the location of Targets to be used by
        the data generator.
        :param valid_file: (str of numpy) the location of validation data set.
        :param target_file: (str of numpy) the location of the validation
        target set.
        :param opt: Optimizer used for training.
        :param samples_per_card: (int) the amount of samples you want passed
        to each gpu card.
        :param epochs: (int) the number of epochs used during training.
        :param gpus_used: (int) amount of GPU cards used to train the
        model.
        :param mode: (str) descriptor of what input data you want to use.
        '2Ch', 'WOFS', or 'FS'.
        :param model_type: (str) descriptor that informas the HP_gridSearch
        of the network type you wish to test. 'UNet' or 'IUNet'
        """
        #self.x_values = D
        #self.x_names = x_names

        #self.rounds = np.prod(self.x_values[:, 3])
        #self.counts = np.zeros(len(self.x_names))

        self.ofolder = ofolder
        #self.top_folder = ofolder + str(self.x_names)
        try:
            os.makedirs(self.o_folder)
        except:
            print('Could not make top folder.')
            #print(self.top_folder)
            print(self.o_folder)
            sys.exit()

        self.batch_folder = batch_folder
        self.target_folder = target_folder
        self.valid_file = valid_file
        self.target_file = target_file
        self.opt = opt

        self.indx = 1

        # Create the dataframe to save results in.
        self.columns = ['Params',
                        'Mean Dice coefficient-BG',
                        'Std. Dev. Dice coefficient-BG',
                        'Mean Dice coefficient-FT',
                        'Std. Dev. Dice coefficient-FT',
                        'Mean Dice coefficient-FGT',
                        'Std. Dev. Dice coefficient-FGT']

        self.evaluation_data_frame = pd.DataFrame(columns=self.columns)

        self.samples_per_card = samples_per_card
        self.epochs = epochs
        self.gpus_used = gpus_used
        self.mode = mode
        self.model_type = model_type

    def basic_test_run(self):


        trainer = Trainer.Trainer(batch_folder=self.batch_folder,
                                  target_folder=self.target_folder,
                                  ofolder=ofolder,
                                  samples_per_card=self.samples_per_card,
                                  epochs=self.epochs,
                                  gpus_used=self.gpus_used,
                                  mode=self.mode,
                                  model_type=self.model_type)

        print('Dilation Rate ' + str(self.x_get_current_value(0)))
        print('Depth ' + str(self.x_get_current_value(1)))
        print('Drop Out ' + str(self.x_get_current_value(2)))

        trainer.train_the_model(t_opt=opt,
                                t_dilRate=self.x_get_current_value(0),
                                t_depth=int(self.x_get_current_value(1)),
                                t_dropOut=self.x_get_current_value(2))

        predictor = Predictor.Predictor(model_folder=ofolder,
                                        batch_file=self.valid_file,
                                        target_file=self.target_file,
                                        ofolder=ofolder,
                                        opt=self.opt)

        results = predictor.predict_and_evaluate()

        print(results)


if __name__ == "__main__":


    # For Oberon
    ofolder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models' \
              '/Temp2/'
    batch_folder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                   '/Batches/2D/Saggital/Training/data/'
    target_folder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                    '/Batches/2D/Saggital/Training/target/'
    validation_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg' \
                      r'-Data/Batches/2D/Saggital/Validdata0.npy'
    target_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                  r'/Batches/2D/Saggital/Validtarget0.npy'

    # opt = k.optimizers.Adam(lr=1e-6)
    opt = k.optimizers.SGD(lr=1e-2, momentum=0.9, decay=0.0, nesterov=True) #SGD as in 2014 paper
    samples_per_card = 256 #i.e. batch size for 1 card --256 as in 2014 paper
    epochs = 75
    gpus_used = 1
    mode = '2Ch'
    model_type = 'VGG'

    a = Run_Basic_Test(ofolder=ofolder,
                       batch_folder=batch_folder,
                       target_folder=target_folder,
                       valid_file=validation_file,
                       target_file=target_file,
                       opt=opt,
                       samples_per_card=samples_per_card,
                       epochs=epochs,
                       gpus_used=gpus_used,
                       mode=mode,
                       model_type=model_type)

    a.basic_test_run()

    print('done')
