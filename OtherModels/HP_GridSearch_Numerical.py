"""
2018-09-30
(c) A. Martel Lab Co.
author: G.Kuling
This is my HyperParameter Grid Search Code for numerical values
WARNING: This file is very hardcoded. Going to need to do some
        editing to generalize it.
"""
import numpy as np
import os
import Trainer
import Predictor as Predictor
import sys
import pandas as pd
import keras as k

np.random.seed(1)

class HP_GridSearch_Numerical:
    def __init__(self,
                 x_names,
                 D,
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
        self.x_values = D
        self.x_names = x_names

        self.rounds = np.prod(self.x_values[:, 3])
        self.counts = np.zeros(len(self.x_names))

        self.ofolder = ofolder
        self.top_folder = ofolder + str(self.x_names)
        try:
            os.makedirs(self.top_folder)
        except:
            print('Could not make top folder.')
            print(self.top_folder)
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

    def token(self, keyword=None):
        """
        Token returns a string identifier for a given grid search point.
        :param keyword: (str) a keyword that can be taked onto the end of the
        token.
        :return: (str)
        """
        token = str('[')

        for i in range(len(self.x_names)):
            token = token + str(self.x_get_current_value(i)) + str(', ')
        token = token + str(']')
        if keyword is not None:
            token = token + str(keyword)
        return token


    def x_get_current_value(self, i):
        """
        Returns the current value of a grid point variable being evaluated.
        :param i: (int) the index of the x value you want to get.
        :return: current x value
        """
        return self.x_values[i, 2] * self.counts[i] + self.x_values[i, 0]

    def increment(self,
                  ver):
        """
        This system increments the hyperparameter grid search parameters
        :param ver:
        :return:
        """
        ver += 1
        if ver == self.indx:
            a = self.indx % np.prod(self.x_values[1:, 3])
            x = 1
            while a != 0:
                x += 1
                a = a % np.prod(self.x_values[x:, 3])
            self.counts[int(x - 1)] += 1
            self.counts[int(x):] = 0

            self.indx += 1
        else:
            print('index and verification mis match... Something went wrong ')
            sys.exit()

    def grid_search_run(self):
        """
        Gird Search Run is the main driver function that executes a full grid
        search. Trains the model with given parameters and saves all the
        training results. Then imports a validation set, predicts on it and save
        the resulting DSC for each validation sample.
        """
        for i in range(int(self.rounds)):
            np.random.seed(1)
            print('Random Seed Reset. ')
            gpt_folder = self.top_folder + '/' + self.token()
            try:
                os.makedirs(gpt_folder)
            except:
                print('Could not make grid point folder.')
                sys.exit()
            trainer = Trainer.Trainer(batch_folder=self.batch_folder,
                                      target_folder=self.target_folder,
                                      ofolder=gpt_folder,
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

            predictor = Predictor.Predictor(model_folder=gpt_folder,
                                            batch_file=self.valid_file,
                                            target_file=self.target_file,
                                            ofolder=gpt_folder,
                                            opt=self.opt)
            results = predictor.predict_and_evaluate()

            error_data_dictionary = \
                {self.columns[0]: self.token(),
                 self.columns[1]: results[0],
                 self.columns[2]: results[1],
                 self.columns[3]: results[2],
                 self.columns[4]: results[3],
                 self.columns[5]: results[4],
                 self.columns[6]: results[5]}
            self.evaluation_data_frame = \
                self.evaluation_data_frame.append(error_data_dictionary,
                                                  ignore_index=True)

            self.evaluation_data_frame.to_csv(
                self.top_folder + '/evaluation_results.csv')

            self.increment(ver=i)


if __name__ == "__main__":
    ### We are going to do a grid search acros x amount of variables with
    # domain D. where d(:,0) are x(i)_min and D(:N) are the x(i)_max.\
    # For Titania
    # ofolder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models' \
    #           '/UNet[SGD(lr=1e-6, momentum=0.9, Nesterov)]-2/'
    # batch_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                '/Batches/2D/Saggital/Training/data/'
    # target_folder = '/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #                 '/Batches/2D/Saggital/Training/target/'
    # validation_file = r'/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg' \
    #                   r'-Data/Batches/2D/Saggital/Validdata0.npy'
    # target_file = r'/labs/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
    #               r'/Batches/2D/Saggital/Validtarget0.npy'

    # For Oberon
    ofolder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data/Models'
    batch_folder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                   '/Batches/2D/Saggital/Training/data/'
    target_folder = '/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                    '/Batches/2D/Saggital/Training/target/'
    validation_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg' \
                      r'-Data/Batches/2D/Saggital/Validdata0.npy'
    target_file = r'/jaylabs/amartel_data2/Grey/2018-10-12-FGTSeg-Data' \
                  r'/Batches/2D/Saggital/Validtarget0.npy'

    # opt = k.optimizers.Adam(lr=1e-6)
    opt = k.optimizers.SGD(lr=1e-6, momentum=0.9, decay=0.0, nesterov=True)
    samples_per_card = 5
    epochs = 10
    #epochs = 50
    #gpus_used = 3
    gpus_used = 1
    mode = '2Ch'
    #model_type = 'UNet'
    model_type = 'VGG'

    x_names = ['t_dilRate', 't_depth', 't_dropOut']
    x_values = np.zeros((len(x_names), 4))
    x_values[0, :] = np.asarray([1, 1, 0, 1])
    x_values[1, :] = np.asarray([7, 7, 0, 1])
    x_values[2, :] = np.asarray([0.25, 0.5, 0.25, 2])

    '''
    a = HP_GridSearch_Numerical(x_names=x_names,
                                D=x_values,
                                ofolder=ofolder,
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
    '''

    a = HP_GridSearch_Numerical(x_names=x_names,
                                D=x_values,
                                ofolder=ofolder,
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

    a.grid_search_run()

    print('done')
