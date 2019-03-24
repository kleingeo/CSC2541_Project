

def ModelParamUtil(model_name):

    assert model_name in ['IUNet', 'UNet', 'VGG', 'ResNet', 'VNet'], \
        'Must select either appropriate model architecture. Must be "IUNet", "UNet", "VNet", "VGG", or "ResNet".'

    if model_name == 'IUNet':

        model_params = {'dilation_rate': 1,
                        'depth': 5,
                        'base_filter': 16,
                        'batch_normalization': False,
                        'pool_1d_size': 2,
                        'deconvolution': False,
                        'dropout': 0.0,
                        'num_classes': 1}

        return model_params

    if model_name == 'UNet':

        model_params = {'dilation_rate': 1,
                        'depth': 5,
                        'base_filter': 16,
                        'batch_normalization': False,
                        'pool_1d_size': 2,
                        'deconvolution': False,
                        'dropout': 0.0,
                        'num_classes': 1}

        return model_params

    if model_name == 'VGG':

        model_params = {'dropout': 0.0,
                        'num_classes': 1}

        return model_params

    if model_name == 'ResNet':

        model_params = {'f': 16,
                        'bn_axis': 3,
                        'dropout': 0.0,
                        'num_classes': 1}

        return model_params

    if model_name == 'VNet':

        model_params = {'dilation_rate': 1,
                        'depth': 5,
                        'base_filter': 16,
                        'batch_normalization': False,
                        'deconvolution': True,
                        'dropout': 0.0,
                        'num_classes': 1}

        return model_params
