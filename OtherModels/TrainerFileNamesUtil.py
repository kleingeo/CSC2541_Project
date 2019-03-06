"""
(C) Martel Lab, 2018

@author: homa
"""
import os
import time
import outils.GridSearch_Consts as GS_Util
import outils.filename_constants as fc


def get_time():
    """
    get the current time , formats it and returns it
    :return: formatted time
    """

    time_stamp = time.strftime("%Y-%m-%d-%H-%M")
    return time_stamp


def create_model_weights_filename(output_directory, time_stamp, keyword):
    """
    this method auto-generates a filename for the weights of the
    trained model , based on the number of epochs, number of epochs per
    batch and time provided by input parameter time_stamp

    :param output_directory: output directory for the weights filename
    :param keyword:the keyword to be used in the filename
    :param time_stamp: this is an string in the format of HH-MM
    """

    model_weights_filename = os.path.join(output_directory,
                                          ('model_weights_' +
                                           keyword + '_' +
                                           time_stamp))
    return model_weights_filename


def create_model_json_filename(output_directory, time_stamp, keyword):
    """
    This method auto-generates a filename for the json file of the model, based
    on the number of epochs and the number of epochs per batch and the time
    provided by the input parameter time_stamp

    :param output_directory: the output directory for the json file
    :param keyword:the keyword to be used in the filename
    :param time_stamp: a string represented a time in the format of HH-MM
    :return:
    """
    model_json_filename = os.path.join(output_directory,
                                       ('model_json_' + keyword +
                                        '_' + time_stamp +
                                        fc.JSON_EXTENSION()))
    return model_json_filename


def create_training_history_filename(output_directory, time_stamp, keyword):
    """
    this method auto-generates a filename for the training history file. The
    filename is based on the number of epochs and number of epochs per batch
    and the time provided by the input parameter, time_stamp

    :param output_directory: the output directory for the json file
    :param keyword:the keyword to be used in the filename
    :param time_stamp: a string in the format of HH-MM
    :return:
    """
    import os
    training_history_filename = os.path.join(output_directory,
                                             ('training_history_' +
                                              keyword + '_' +
                                              time_stamp +
                                              fc.CSV_EXTENSION()))
    return training_history_filename


def build_filename_keyword(model_param):
    keyword = ('op_' + str(model_param[GS_Util.OPTIMIZERS()]) + '_dr_' +
               str(model_param[GS_Util.DILATION_RATE()]) + '_depth_' +
               str(model_param[GS_Util.DEPTH()]) + '_dropout_' +
               str(model_param[GS_Util.DROPOUT()]) + '_kernel1dsize_' +
               str(model_param[GS_Util.KERNEL_1D_SIZE()]))
    return keyword


def create_output_directory(directory, time_stamp):
    """
    creats a subdirectory in the directory for all the model's related files
    to be saved in
    :param directory: the top level output directory
    :param time_stamp: the sub folder has the same name as the time stamp
    :return:
    """

    sub_directory = os.path.join(directory, time_stamp)
    if not os.path.exists(sub_directory):
        os.makedirs(sub_directory)
        return sub_directory
    else:
        raise ValueError(sub_directory + ' exists')
