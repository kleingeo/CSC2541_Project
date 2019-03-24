"""
(C) Martel Lab, 2018

@author: homa
"""
import os
import time
import ModelTraining.GridSearch_Consts as GS_Util


def get_time():
    """
    get the current time , formats it and returns it
    :return: formatted time
    """

    time_stamp = time.strftime("%Y-%m-%d-%H-%M")
    return time_stamp


def create_model_weights_filename(output_directory , keyword):
    """
    this method auto-generates a filename for the weights of the
    trained model , based on the number of epochs, number of epochs per
    batch and time provided by input parameter time_stamp

    :param output_directory: output directory for the weights filename
    :param keyword:the keyword to be used in the filename
    """

    model_weights_filename = os.path.join(output_directory, keyword)
    return model_weights_filename


def create_model_json_filename(output_directory, keyword):
    """
    This method auto-generates a filename for the json file of the model, based
    on the number of epochs and the number of epochs per batch and the time
    provided by the input parameter time_stamp

    :param output_directory: the output directory for the json file
    :param keyword:the keyword to be used in the filename
    :return:
    """
    model_json_filename = os.path.join(output_directory,
                                       ('model_json_' + keyword + '.json'))
    return model_json_filename


def create_training_history_filename(output_directory, keyword):
    """
    this method auto-generates a filename for the training history file. The
    filename is based on the number of epochs and number of epochs per batch
    and the time provided by the input parameter, time_stamp

    :param output_directory: the output directory for the json file
    :param keyword:the keyword to be used in the filename
    :return:
    """
    import os
    training_history_filename = os.path.join(output_directory,
                                             ('training_history_' + keyword + '.csv'))
    return training_history_filename


def build_filename_keyword(model_param):


    if model_param[GS_Util.TRAIN_FRAC()] == 1:
        train_frac = str(int(model_param[GS_Util.TRAIN_FRAC()]))

    else:
        train_frac = str(model_param[GS_Util.TRAIN_FRAC()]).replace('.', '')

    keyword = (str(model_param[GS_Util.MODEL_TYPE()]) + '_' +
               train_frac + '_' +
               str(model_param[GS_Util.WITH_FAKE()]) + '_' +
               str(model_param[GS_Util.AUGMENT_TRAINING()])
               )

    return keyword


def create_output_directory(directory, time_stamp=None):
    """
    creats a subdirectory in the directory for all the model's related files
    to be saved in
    :param directory: the top level output directory
    :param time_stamp: the sub folder has the same name as the time stamp
    :return:
    """

    if time_stamp is not None:
        sub_directory = os.path.join(directory, time_stamp)
    else:
        sub_directory = directory

    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
        return sub_directory
    else:
        raise ValueError(sub_directory + ' exists')
