'''
2018-09-28
(c) A.Martel Lab Co.
suthor: G.Kuling
'''
import numpy as np
import os
import pandas as pd
from keras import backend as K


def dice(y_true, y_pred):
    """This method calculates the dice coefficient between the true
     and predicted masks
    Args:
        y_true: The true mask(i.e. ground-truth or expert annotated mask)
        y_pred: The predicted mask

    Returns:
        double: The dice coefficient"""
    smooth = K.epsilon()
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f)
                                           + smooth)

def dice_loss(y_true, y_pred):
    '''
    dice coefficient loss function.
    :param y_true: The true mask(i.e. ground-truth or expert annotated mask)
    :param y_pred: The predicted mask
    :return: negative of the dice_loss
    '''
    return -dice(y_true, y_pred)