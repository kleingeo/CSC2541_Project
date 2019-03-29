from ModelPredictor.Predictor import Predictor
import pandas as pd
import os

if __name__ == '__main__':

    df = pd.read_pickle('../Dataset/seg_slice_dataframe.pickle')

    t2_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_T1'
    seg_file_path = '/localdisk1/GeoffKlein/BRATS2018/MICCAI_BraTS_2018_Data_Training/HGG'
    t1_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_T1'
    flair_file_path = '/localdisk1/GeoffKlein/BRATS2018/T2_Flair'

    # t2_file_path = 'D:/Geoff_KLein/BRATS2018/T2_T1'
    # seg_file_path = 'D:/Geoff_KLein/BRATS2018/MICCAI_BraTS_2018_Data_Training/HGG'
    # t1_file_path = 'D:/Geoff_KLein/BRATS2018/T2_T1'
    # flair_file_path = 'D:/Geoff_KLein/BRATS2018/T2_Flair'


    sample_main_path = {'t2': t2_file_path,
                        't1': t1_file_path,
                        'flair': flair_file_path,
                        'seg': seg_file_path}


    df_testing = df.loc[df['train_val_test'] == 'val']


    df_total_eval_dsc = None

    top_output_directory = '../TrainOutput'

    time_stamp = '2019-03-27-15-51'



    for idx, dir in enumerate(os.listdir(top_output_directory + '/' + time_stamp)):


        if (dir != 'UNet_1_False_True'):
            continue


        if os.path.isfile(top_output_directory + '/' + time_stamp + '/' + dir):
            continue


        model_output_directory = dir




        weights_folder_path = top_output_directory + '/' + time_stamp + '/' + dir



        for weights in os.listdir(weights_folder_path):

            if weights.endswith('_weights.h5') is False:

                continue

            if weights.endswith('100_weights.h5') is False:
                continue

            model_weights_filename = weights_folder_path + '/' + weights

            model_json_filename = weights_folder_path + '/' + 'model_json_' + dir




            if weights == 'vnet_adam_1_6_3_32_0_True_None_200_weights.h5':
                save_predictions = False

            else:
                save_predictions = False


            pred = Predictor(model_weights_filename=model_weights_filename,
                             model_json_filename=model_json_filename + '.json',
                             testing_sample_dataframe=df_testing,
                             main_sample_directory=sample_main_path,
                             main_output_directory=top_output_directory,
                             model_output_directory=model_output_directory,
                             save_predictions=True,
                             time_stamp=time_stamp)

            if os.path.exists(pred.evaluation_filename + '.pickle'):

                df_eval = pd.read_pickle(pred.evaluation_filename + '.pickle')


            else:

                pred.predict_and_evaluate()

                df_eval = pred.measures_df


            # Initialize dataframe of all evaluations
            if df_total_eval_dsc is None:

                df_total_eval_dsc = df_eval[['test_file', 'ground_truth']]


            df_total_eval_dsc = pd.concat([df_total_eval_dsc, df_eval['dice']], axis=1)

            column_name = weights.split('_weights')[0]

            df_total_eval_dsc = df_total_eval_dsc.rename(columns={"dice": column_name})


    # df_total_eval_dsc.to_pickle(top_output_directory + '/' + time_stamp + '/' + 'validation_dice_scores.pickle')
    #
    # df_total_eval_dsc.to_csv(top_output_directory + '/' + time_stamp + '/' + 'validation_dice_scores.csv', index=False)
