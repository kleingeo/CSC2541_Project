"""
(c) Anne Martel Lab, 2018
created: 05/17/2018
Author: Grey Kuling
Refactored by: Homa
"""

import os

import numpy as np
import pandas as pd

# import logs.config_initialization as config_initialization
from StatTests.stat_tests import kw_dunn


class KWDunnTestForGS:

    def __init__(self, val_path):

        self.val_path = val_path

    def load_datframes(self):

        for idx, path in enumerate(self.val_path):

            if idx == 0:

                df_main = pd.read_pickle(path + '/validation_concurrancy_scores.pickle')

                continue

            df_hold = pd.read_pickle(path + '/validation_concurrancy_scores.pickle')
            df_hold = df_hold.drop(['test_file', 'ground_truth', 'dataframe_idx'], axis=1)

            df_main = pd.concat([df_main, df_hold], axis=1)


        df_main = df_main.drop(['test_file', 'ground_truth', 'dataframe_idx'], axis=1)

        column_names = df_main.columns.tolist()

        # val_score_matrix = df_main.values
        #
        # df_main = df_main

        self.column_names = column_names
        self.df_main = df_main

    def perform_test(self):
        """
        Executes the KW test with Dunn's multiple comparison test.
        :return: saves a csv file of the results
        """
        self.load_datframes()

        H, p_omnibus, Z_pairs, p_corrected, reject = \
            kw_dunn([self.df_main[col] for col in self.df_main.columns],
                    to_compare=None,
                    alpha=0.05, method='bonf')

        group_1 = []
        group_2 = []
        for i in range(len(self.column_names)):
            for j in range(len(self.column_names) - i - 1):
                print('Testing model ' + str(i) + ' and ' + str(j))
                group_1.append(self.column_names[i])
                group_2.append(self.column_names[i + j + 1])

        d = {'Group 1': group_1,
             'Group 2': group_2,
             'Z pairs': Z_pairs,
             'p_corrected': p_corrected,
             'Reject H0': reject,
             'H Stat': H,
             'p_value': p_omnibus
             }

        self.stat_results = pd.DataFrame(data=d)

    def save_stat_file(self, filename):
        """
        saves the stat values to fi
        :param df: the data frame
        :return:
        """

        try:
            self.stat_results.to_csv(filename, index=False)
            print('The stat values is saved successfully to' + filename)
        except IOError:
            print('Can not save the stat values to ' + filename)

    def save_cumulative_val_dsc(self, filename=None):

        median_series = self.df_main.median(axis=0)
        mean_series = self.df_main.mean(axis=0)
        std_series = self.df_main.std(axis=0)

        val_scores_df = pd.concat([median_series, mean_series, std_series], axis=1)

        val_scores_df = val_scores_df.reset_index()

        val_scores_df = val_scores_df.rename(columns={'index': 'model',
                                                      0: 'median_dsc',
                                                      1: 'mean_dsc',
                                                      2: 'std_dsc'})

        if filename is not None:
            val_scores_df.to_csv(filename, index=False)

        self.val_scores_df = val_scores_df


    def determine_most_sig(self):

        self.perform_test()
        self.save_cumulative_val_dsc()

        self.val_scores_df = self.val_scores_df.sort_values('mean_dsc', ascending=False)

        total_reject_h0 = []

        for idx, row in self.val_scores_df.iterrows():

            model_name = row['model']

            df_hold = self.stat_results.loc[((self.stat_results['Group 1'] == model_name) |
                                             (self.stat_results['Group 2'] == model_name))]

            reject_h0 = df_hold['Reject H0'].values

            total_reject_h0.append(np.sum(reject_h0))

        return



if __name__ == '__main__':

    main_df_path = '../Model/SpineSeg_08_08_125_model/VBnoCortSeg_Registered'

    validation_df_root = [main_df_path + '/2018-11-28-14-14',
                          main_df_path + '/2018-11-30-12-33',
                          main_df_path + '/2018-12-01-23-28',
                          main_df_path + '/2018-12-02-10-24',
                          main_df_path + '/2018-12-03-08-55',
                          main_df_path + '/2018-12-03-21-16',
                          main_df_path + '/2018-12-04-17-09']


    # validation_df_root = [main_df_path + '/2018-11-28-14-14',
    #                       main_df_path + '/2018-12-01-23-28']

    kw = KWDunnTestForGS(validation_df_root)

    # kw.perform_test()

    kw.determine_most_sig()

    # kw.save_cumulative_val_dsc('val_dsc.csv')
    # kw.save_stat_file('compare_models_kw.csv')
