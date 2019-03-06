
import SimpleITK as sitk


def cubic_resample_new(image, spacing):
    """Resample volume to cubic voxels using sitk routines

    Args:
        image: sitk image
        spacing:

    Returns:
        sitk image

     """
    resample = sitk.ResampleImageFilter()

    spacingOut = spacing
    resample.SetOutputSpacing(spacingOut)
    spacingIn = image.GetSpacing()
    shape = image.GetSize()

    newSize = [round(shape[0] * spacingIn[0] / spacingOut[0]),
               round(shape[1] * spacingIn[1] / spacingOut[1])]
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetSize(newSize)
    new = resample.Execute(image)
    new.SetDirection(image.GetDirection())
    return new
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
    if mode == '2Ch':
        return (2, [0,1])
    if mode =='WOFS':
        return (1, [1])
    if mode == 'FS':
        return (1, [0])

def My_new_loss(y_true, y_pred):

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

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)