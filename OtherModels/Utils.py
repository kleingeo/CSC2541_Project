'''
2018-09-28
(c) A.Martel Lab Co.
suthor: G.Kuling
'''
import numpy as np
import os
import pandas as pd
from keras import backend as K

def decide_chnls(mode):
    '''
    Helps to determine dimensionality of modality choice.
    :param mode: (str) choice of imaging modality. '2Ch', 'WOFS', or 'FS'
    :return: (tuple) number of channels, list of channel indices.
    '''
    if mode == '2Ch':
        return (2, [0,1])
    if mode =='WOFS':
        return (1, [1])
    if mode == 'FS':
        return (1, [0])

def My_new_loss(y_true, y_pred):
    '''
    My Troversky loss function.
    :param y_true:
    :param y_pred:
    :return:
    '''

    ave_DSC = (dice_coef(y_true[:, 0], y_pred[:, 0]) +
               dice_coef(y_true[:,1] ,y_pred[:,1]) +
               dice_coef(y_true[:, 2], y_pred[:, 2]))

    return 3.0 - ave_DSC

def dice_coef(y_true, y_pred):
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
    return -dice_coef(y_true, y_pred)