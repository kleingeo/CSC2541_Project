"""
(c) Martel Lab , 2018
Created on Jan 23, 2018

@author: homa

"""
import os
import numpy as np
import pandas as pd
from keras.models import model_from_json


from OtherModels.Utils import dice_loss, dice


from log.logging_dict_configuration import logging_dict_config
import logging
from logging.config import dictConfig

# from UNetScripts.TrainerPredictorUtils import TrainerPredictorUtils
# import Util.filename_constants as FileConstUtil

from Dataset.DataGenerator import DataGenerator


class Predictor:
    """this class uses a pre-trained model and evaluates the model on the
    test set. It also predicts the outputs and
    """

    def __init__(self,
                 model_weights_filename,
                 model_json_filename,
                 sample_size,
                 testing_sample_dataframe,
                 main_sample_directory,
                 main_output_directory,
                 model_output_directory,
                 time_stamp=None,
                 data_frame_row=None,
                 save_predictions=False,
                 predicting_log_name=None):
        """
        This is the constructor of the Predictor class

        :param model_weights_filename: the full file name of the model
            weigths(h5 file)
        :param model_json_filename: the full filename of the model json file
        :param input_directory: the test batches live here
        :param output_directory: the predicted masks and the evaluation csv file
            will be saved here
        """
        # assignments


        self.model_weights_filename = model_weights_filename
        self.model_json_filename = model_json_filename
        self.data_frame_row = data_frame_row
        self.main_output_directory = main_output_directory
        self.model_output_directory = model_output_directory

        self.testing_sample_dataframe = testing_sample_dataframe

        self.main_sample_directory = main_sample_directory
        self.sample_size = sample_size
        self.save_predictions = save_predictions
        self.time_stamp = time_stamp


        #
        # self.time_stamp = self.model_weights_filename.split('/')[0]

        self.predicting_log_name = predicting_log_name



        if self.predicting_log_name is None:

            self.predicting_log_name = self.main_output_directory + '/' + self.time_stamp + '/' + self.model_output_directory + '/' + 'predict.log'

        self.logging_dict = logging_dict_config(self.predicting_log_name)

        dictConfig(self.logging_dict)

        self.logger = logging.getLogger(__name__)

        self.logger_format = logging.Formatter(fmt=self.logging_dict['formatters']['f']['format'],
                                               datefmt=self.logging_dict['formatters']['f']['datefmt'])




        self.create_evaluation_filename()
        self.logger.info('evaluation_filename is ' + self.evaluation_filename)


    def load_model(self):
        """
        this method loads the pre-trained model in two steps
        first it loads the json file and then it loads the weights. The
        compilation parameters are also set, which are the same as the
        training time
        """

        self.logger.info('loading the model')

        # load json file first

        json_file = open(self.model_json_filename, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # now load weights

        self.model.load_weights(self.model_weights_filename)
        # set the compilation parameters

        self.model.compile(optimizer='Adam',
                           loss=dice_loss,
                           metrics=[dice])

        self.logger.info('model loaded')

    def create_predicted_batch_filename(self, test_batch_filename, predictions_test_target):
        """
        this method generates a filename for the predictions to be saved to
        the filename is built based on the model_weights_filename and
        test_batch_filename. The file is located at the output_directory

        :param test_batch_filename: this is the filename of the test batch

        """

        base_filename = test_batch_filename.split('.')[0]

        base_weight_filename = self.extract_base_filename(self.model_weights_filename)


        pred_batch_main_path = os.path.join(self.main_output_directory, self.time_stamp,
                                            self.model_output_directory, 'prediction')

        if os.path.exists(pred_batch_main_path) is False:

            os.makedirs(pred_batch_main_path)


        pred_batch_filename = pred_batch_main_path + '/' + base_filename + predictions_test_target\
                              + '.npy'


        return pred_batch_filename

    def predict_and_evaluate(self):
        """
        this method evaluates the pre-trained model on the batches of the
        test images, the ground truth masks are also available.
        the evaluations on the batches are all saved in the a file.
        For more information on the name of the evaluation file, refer to
        self.create_evaluation_filename
        """

        self.model = None
        self.load_model()

        # take care of evaluations and evaluation filename


        self.measures_df = pd.DataFrame(
            columns=['test_file'] + ['ground_truth'] + self.model.metrics_names)


        # load the test data
        self.logger.info('evaluation started')

        t2_img_test = self.testing_sample_dataframe['t2_filename'].tolist()
        t1_img_test = self.testing_sample_dataframe['t1_filename'].tolist()
        flair_img_test = self.testing_sample_dataframe['flair_filename'].tolist()


        seg_test = self.testing_sample_dataframe['seg_filename'].tolist()
        seg_slice = self.testing_sample_dataframe['slice_number'].tolist()

        t2_file_path = self.main_sample_directory['t2']
        t1_file_path = self.main_sample_directory['t1']
        flair_file_path = self.main_sample_directory['flair']

        seg_file_path = self.main_sample_directory['seg']


        params_test_gen = {'sample_size': self.sample_size,
                           'batch_size': 1,
                           'n_channels': 3,
                           'shuffle': False,
                           'augment_data': False}

        test_gen = DataGenerator(
            t2_sample=t2_img_test,
            seg_sample=seg_test,
            t2_sample_main_path=t2_file_path,
            seg_sample_main_paths=seg_file_path,
            seg_slice_list=seg_slice,
            t1_sample=t1_img_test,
            flair_sample=flair_img_test,
            t1_sample_main_path=t1_file_path,
            flair_sample_main_path=flair_file_path,
            **params_test_gen)

        self.model.

        for idx in range(len(t2_img_test)):

            print(('loading batch ' + str(idx) + ' of '
                   + str(len(x_test))))


            test_images, ground_truth_masks = test_gen.__getitem__(idx)



            self.logger.info('Prediction and evaluating batch ' + str(idx))


            if test_images is not None and ground_truth_masks is not None:

                results = self.model.test_on_batch(test_images, ground_truth_masks)

                dict_hold = {'test_file': x_test[idx],
                             'ground_truth': y_test[idx],
                             'dataframe_idx': df_index_list[idx]}

                for idx2 in range(len(self.model.metrics_names)):

                    dict_hold[self.model.metrics_names[idx2]] = results[idx2]

                self.measures_df = self.measures_df.append(dict_hold, ignore_index=True)


            if (test_images is not None) and (self.save_predictions is True):

                test_filename = self.create_predicted_batch_filename(x_test[idx], FileConstUtil.TEST_DATA())

                ground_truth_filename = self.create_predicted_batch_filename(y_test[idx], FileConstUtil.TEST_DATA_TARGET())

                predicted_filename = self.create_predicted_batch_filename(y_test[idx], FileConstUtil.PREDICTION())


                predicted_masks = self.model.predict_on_batch(test_images)

                self.logger.info('Saving the predictions to ' + predicted_filename)

                predicted_masks2 = np.copy(predicted_masks)

                predicted_masks2[predicted_masks > 0.5] = 1
                predicted_masks2[predicted_masks <= 0.5] = 0

                predicted_masks_img = sitk.GetImageFromArray(predicted_masks2[0, :, :, :, 0])

                ground_truth_img = sitk.GetImageFromArray(ground_truth_masks[0, :, :, :, 0])

                test_img = sitk.GetImageFromArray(test_images[0, :, :, :, 0])


                sitk.WriteImage(predicted_masks_img, predicted_filename)
                sitk.WriteImage(ground_truth_img, ground_truth_filename)
                sitk.WriteImage(test_img, test_filename)

                self.logger.info(
                'Evaluation completed on batch ' + str(idx))


        if self.save_predictions is False:

            self.logger.info('Predictions were not saved as save_predictions = ' + str(self.save_predictions))

        self.save_evaluation_file()


    def create_evaluation_filename(self):
        """
        this method creates a filename for the evaluation results.
        the file is located in the self.output_directory and it is
        built based on the name of the self.model_weights_filename

        Example:
            if the following values are set for the output_directory and the
            model_weights_filename,
            self.output_directory = 'd:/output/'
            self.model_weights_filename = 'd:/weights/weights123.h5'
            then, the self.evaluation_filename will be
            'd:/output/weights123.csv'
        """

        base_filename = self.extract_base_filename(self.model_weights_filename)
        # create a csv filename inside the output_directory
        self.evaluation_filename = os.path.join(
            self.main_output_directory,
            self.time_stamp,
            self.model_output_directory,
            ('evaluation_' + base_filename))


    def extract_base_filename(self, filename):
        # get the filename
        base_filename = os.path.basename(filename)
        # remove the extension
        base_filename = os.path.splitext(base_filename)[0].split('_weights')[0]
        return str(base_filename)

    def save_evaluation_file(self):
        """
        this method saves the results of the evaluation into a csv file
        """
        try:
            self.measures_df.to_csv(self.evaluation_filename + '.csv', index=False)
            self.measures_df.to_pickle(self.evaluation_filename + '.pickle')

            self.logger.info(
                'evaluation measures are saved to:' + self.evaluation_filename)
        except:
            self.logger.error(
                'can not save evaluation measures to: ' +
                self.evaluation_filename)



if __name__ == "__main__":
    import sys

    logger = config_initialization.logging.getLogger(__name__)
    parser = get_parser()
    try:
        args = parser.parse_args()
        logger.info(args)
        main_driver(args.json_file,
                    args.weights_file,
                    args.batches_dir,
                    args.output_dir,
                    args.batches_dir_type,
                    args.optimizer_type
                    )
    except ArgumentError as arg_exception:
        logger.error("Argument Error: {0}".format(arg_exception))
    except Exception as exception:
        logger.error("Exception: {0}".format(exception))

    sys.exit()
