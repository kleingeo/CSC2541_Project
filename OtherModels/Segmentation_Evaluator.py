'''
2018-12-12
(c) A.Martel Lab Co.
Author: G.Kuling
This file takes in a BM and FGT ground truth and evaluates metrics compared
to the predicted tissues of the network.
'''

import SimpleITK as sitk
import numpy as np
import Segmenter
import Evaluation_Measures as em

def give_me_metrics(FGT, BM, tissues):
    '''
    Takes in ground truth and predicted segmentations and evaluates:
        DSC of Fat
        DSC of FGT
        Jaccard of Fat
        Jaccard of FGT
        Relative Absolute volume difference of Fat
        Relative Absolute volume difference of FGT
        Volume of Fat GT
        Volume of FGT GT
        Volume of Fat PRED
        Volume of FGT PRED
    For the Right and Left side of the scan.
    :param FGT: (SITK Image) of FGT ground truth
    :param BM: (SITK IMAGE) of Breast MAsk Ground Truth
    :param tissues: (SITK Image) of tissue segmentations (1 for Fat, 2 for FGT)
    :return: Results are printed on the screen. No Return Results.
    '''

    ### Separate volumes
    fgt = sitk.GetArrayFromImage(FGT)

    fat = sitk.GetArrayFromImage(BM) - fgt

    if fat.max !=1:
        inds = np.where(fat > 0)
        fat[inds] = 1

    p_fgt = sitk.GetArrayFromImage(tissues) >= 2
    p_fgt = p_fgt.astype('float32')

    p_fat = sitk.GetArrayFromImage(tissues) == 1
    p_fat = p_fat.astype('float32')

    ## begin evaluations
    mid_pt = int(fgt.shape[0]/2)

    print('Right Breast Side: ')
    print('DSC for Fat Tissue: ' +
          str(em.dice_coefficient_numpy_arrays(fat[:mid_pt,...],
                                               p_fat[:mid_pt,...])))
    print('DSC for FGT Tissue: '
          + str(em.dice_coefficient_numpy_arrays(fgt[:mid_pt,...],
                                                 p_fgt[:mid_pt,...])))

    print('JAC for Fat Tissue: ' +
          str(em.jaccard_coefficient(fat[:mid_pt,...], p_fat[:mid_pt,...])))
    print('JAC for FGT Tissue: ' +
          str(em.jaccard_coefficient(fgt[:mid_pt,...], p_fgt[:mid_pt,...])))

    print('RAVD for Fat Tissue: ' + str(
        em.relative_absolute_volume_difference(fat[:mid_pt,...],
                                               p_fat[:mid_pt,...])))
    print('RAVD for FGT Tissue: ' + str(
        em.relative_absolute_volume_difference(fgt[:mid_pt,...],
                                               p_fgt[:mid_pt,...])))

    print('Vol. Predicted for Fat Tissue: ' + str(np.sum(p_fat[:mid_pt,...])))
    print('Vol. Predicted for FGT Tissue: ' + str(np.sum(p_fgt[:mid_pt,...])))

    print('Vol. GT for Fat Tissue: ' + str(np.sum(fat[:mid_pt,...])))
    print('Vol. GT for FGT Tissue: ' + str(np.sum(fgt[:mid_pt,...])))

    print('Left Breast Side: ')
    print('DSC for Fat Tissue: ' +
          str(em.dice_coefficient_numpy_arrays(fat[mid_pt:, ...],
                                               p_fat[mid_pt:, ...])))
    print('DSC for FGT Tissue: '
          + str(em.dice_coefficient_numpy_arrays(fgt[mid_pt:, ...],
                                                 p_fgt[mid_pt:, ...])))

    print('JAC for Fat Tissue: ' +
          str(em.jaccard_coefficient(fat[mid_pt:, ...], p_fat[mid_pt:, ...])))
    print('JAC for FGT Tissue: ' +
          str(em.jaccard_coefficient(fgt[mid_pt:, ...], p_fgt[mid_pt:, ...])))

    print('RAVD for Fat Tissue: ' + str(
        em.relative_absolute_volume_difference(fat[mid_pt:, ...],
                                               p_fat[mid_pt:, ...])))
    print('RAVD for FGT Tissue: ' + str(
        em.relative_absolute_volume_difference(fgt[mid_pt:, ...],
                                               p_fgt[mid_pt:, ...])))

    print('Vol. Predicted for Fat Tissue: ' + str(np.sum(p_fat[mid_pt:, ...])))
    print('Vol. Predicted for FGT Tissue: ' + str(np.sum(p_fgt[mid_pt:, ...])))

    print('Vol. GT for Fat Tissue: ' + str(np.sum(fat[mid_pt:, ...])))
    print('Vol. GT for FGT Tissue: ' + str(np.sum(fgt[mid_pt:, ...])))
    print('done')


if __name__ == "__main__":
    scan_local = [r'Y:\Hongbo\processed_data\23_0132_5154279_wofs.mha',
                  r'Y:\Hongbo\processed_data\23_0132_5154279_800@094223.mha']

    FGT = sitk.ReadImage(r'Y:\Grey\2018-10-12-FGTSeg-Data\FGT_masks' \
                         r'\23_0132_5154279_FGTmask.mha')

    BM = sitk.ReadImage(r'Y:\Grey\2018-10-12-FGTSeg-Data\CNN_breastmasks'
                        r'\23_0132_5154279_wofs_CNNBreastmask.mha')

    model_dir = r'X:\2018-09-28-FGTSeg2\model\\'

    ofolder = r'Y:\Grey\2018-10-12-FGTSeg-Data\Segmented\\'

    a = Segmenter.Segmenter(scan_local, model_dir, ofolder)

    tissues = a.perform_segmentation()

    a.save_segmentations()

    # tissues = sitk.ReadImage(
    #     r'Y:\Grey\2018-10-12-FGTSeg-Data\Segmented\141_0685_5456684_TissueSeg'
    #     r'.mha')
    give_me_metrics(FGT, BM, tissues)


