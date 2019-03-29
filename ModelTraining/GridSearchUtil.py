import pandas as pd
import ModelTraining.GridSearch_Consts as Util


class GridSearchUtil:
    """this class builds the hyper-parameter search space"""
    def __init__(self, params_dictionary):

        """
        constructor
        :param params_dictionary: this is not None, if the user wants to
        train the network based on one set of arguments.
        """

        self.build_grid_search_params_from_input(params_dictionary)


    def save(self, output_filename=None, output_location=None):
        """
        saves the data-frame to a csv file
        :param output_filename: the output filename
        :return:
        """

        if (output_location is not None) and (output_filename is None):

            self.params_dataframe.to_csv(output_location + '/grid_search.csv')

        elif (output_location is None) and (output_filename is None):
            self.params_dataframe.to_csv('grid_search.csv')

        elif (output_location is None) and (output_filename is not None):
            self.params_dataframe.to_csv(output_filename)

        elif (output_location is not None) and (output_filename is not None):
            self.params_dataframe.to_csv(output_location + '/' + output_filename)

    def build_grid_search_params_from_input(self, params_dictionary):
        """
        this method builds a data frame with one row, based on the
        user input parameters
        :param params_dictionary: the input parameters
        :return:
        """

        from itertools import product

        default_dictionary = dict(model_type=self.get_model_type(),
                                  with_fake=self.get_with_fake(),
                                  train_faction=self.get_train_fraction(),
                                  epochs=self.get_epochs(),
                                  batch_size=self.get_batch_sizes(),
                                  augment_training=self.get_augment_training(),
                                  )

        if Util.MODEL_TYPE() in params_dictionary.keys():
            default_dictionary[Util.MODEL_TYPE()] = params_dictionary[Util.MODEL_TYPE()]

        if Util.WITH_FAKE() in params_dictionary.keys():
            default_dictionary[Util.WITH_FAKE()] = params_dictionary[Util.WITH_FAKE()]

        if Util.TRAIN_FRAC() in params_dictionary.keys():
            default_dictionary[Util.TRAIN_FRAC()] = params_dictionary[Util.TRAIN_FRAC()]

        if Util.EPOCHS() in params_dictionary.keys():
            default_dictionary[Util.EPOCHS()] = params_dictionary[Util.EPOCHS()]

        if Util.BATCH_SIZE() in params_dictionary.keys():
            default_dictionary[Util.BATCH_SIZE()] = params_dictionary[Util.BATCH_SIZE()]

        if Util.AUGMENT_TRAINING() in params_dictionary.keys():
            default_dictionary[Util.AUGMENT_TRAINING()] = params_dictionary[Util.AUGMENT_TRAINING()]


        permuted_dictionary = [dict(zip(default_dictionary, v)) for v in
                               product(*default_dictionary.values())]

        self.params_dataframe = pd.DataFrame(permuted_dictionary)

    def get_params_dataframe(self):
        """

        :return: returns the hyper parameters data frame
        """
        return self.params_dataframe


    @staticmethod
    def get_model_type():
        """
        Get the correct architecture to use to build the model
        :return: list of available architectures
        """
        model_type = ['UNet']
        return model_type


    @staticmethod
    def get_train_fraction():
        """
        Get the fraction for training data
        :return: list of available architectures
        """
        train_frac = [1]
        return train_frac


    @staticmethod
    def get_with_fake():
        """
        Determine if to use the fake data from the GAN during training.

        :return: list of available architectures
        """
        with_fake = [False]
        return with_fake

    @staticmethod
    def get_epochs():
        """
        number of epochs
        :return: list of epochs
        """
        epochs = [100]
        return epochs


    @staticmethod
    def get_batch_sizes():
        batch_sizes = [5]
        return batch_sizes


    @staticmethod
    def get_augment_training():
        """
        whether training data is augmented
        :return:
        """
        augment_training = [True]
        return augment_training


if __name__ == "__main__":
    '''
    for demo purposes only
    '''
    grid_searcher_util = GridSearchUtil()
    dataframe = grid_searcher_util.get_params_dataframe()
    print(dataframe)
    # grid_searcher_util.save()
