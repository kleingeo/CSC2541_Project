# import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc

import DGenerator as generator


def plot_avg_auc(testing_prob_maps_path, testing_ground_truth_path, testing_img_path):

    y_decision_score = np.load(testing_prob_maps_path)

    gen = generator.DGenerator(data_dir=testing_img_path,
                               target_dir=testing_ground_truth_path,
                               batch_size=1,
                               regular=True,
                               shuffle=False)

    gen.batch_size = gen.__len__()

    testing_ground_truth_img, testing_ground_truth = gen.__getitem__(0)

    gt_hold = testing_ground_truth[:, 1, :, :]

    pred_roc = y_decision_score[:, 1, :, :]

    fpr = []
    tpr = []


    n_threshold = 200

    threshold_vec = np.linspace(0, 1, n_threshold)

    for idx in range(gt_hold.shape[0]):


        gt = testing_ground_truth[idx, 1, :, :].flatten()

        y_prob = y_decision_score[idx, 1, :, :].flatten()

        # y_prob_threshold = np.copy(y_prob)

        pos_idx = np.where(gt == 1)[0]

        neg_idx = np.where(gt == 0)[0]

        fpr_hold_sample = []

        tpr_hold_sample = []

        for threshold in threshold_vec:


            # y_prob_threshold[y_prob > threshold] = 1
            # y_prob_threshold[y_prob < threshold] = 0

            y_prob_threshold = np.where(y_prob >= threshold, 1, 0)

            # tn0, fp0, fn0, tp0 = sklearn.metrics.confusion_matrix(gt, y_prob_threshold).ravel()

            tp = (y_prob_threshold[pos_idx] == 1).astype(int).sum()
            fp = (y_prob_threshold[neg_idx] == 1).astype(int).sum()

            tn = (y_prob_threshold[neg_idx] == 0).astype(int).sum()
            fn = (y_prob_threshold[pos_idx] == 0).astype(int).sum()


            tpr_hold = tp / (tp + fn)
            fpr_hold = fp / (tn + fp)

            # tpr_hold = tp0 / (fp0 + tp0)
            # fpr_hold = fp0 / (fp0 + tp0)

            fpr_hold_sample.append(fpr_hold)
            tpr_hold_sample.append(tpr_hold)


        fpr_hold_sample = np.array(fpr_hold_sample)
        tpr_hold_sample = np.array(tpr_hold_sample)


        fpr.append(fpr_hold_sample)
        tpr.append(tpr_hold_sample)

    fpr = np.stack(fpr, axis=-1)
    tpr = np.stack(tpr, axis=-1)

    fpr_avg = np.mean(fpr, axis=1)
    tpr_avg = np.mean(tpr, axis=1)

    roc_auc = auc(fpr_avg, tpr_avg)

    # f1_metric = sklearn.metrics.f1_score(y_test, y_pred)

    plt.figure()
    plt.plot(fpr_avg, tpr_avg, color='red',
             lw=1, label='ROC curve (Area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC For Prostate Segmentation - Augmented UNet')
    plt.legend(loc="lower right")
    plt.show()

    print()

    plt.savefig('augmented_regular_unet_roc.png', bbox_inches='tight', dpi=650)

if __name__ == '__main__':

    testing_prob_maps_path = 'UNet_regularWAugcGAN_grey2/test_results/pred_TestSet.npy'

    testing_ground_truth_path = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/labelsTs/'

    testing_img_path = 'D:/Geoff_Klein/Prostate_Data/Task05_Prostate/imagesTs/'

    plot_avg_auc(testing_prob_maps_path, testing_ground_truth_path, testing_img_path)