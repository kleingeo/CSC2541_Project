import OtherModels.IUNet as IUNet
import OtherModels.UNet as UNet
import OtherModels.VGG as VGG
import OtherModels.ResNet as ResNet
import OtherModels.VNet as VNet

def ModelSelectUtil(model_name):


    assert model_name in ['IUNet', 'UNet', 'VGG', 'ResNet', 'VNet'], \
        'Must select either appropriate model architecture. Must be "IUNet", "UNet", "VNet", "VGG", or "ResNet".'

    if model_name == 'IUNet':

        model_fn = IUNet.get_iunet

        return model_fn

    if model_name == 'UNet':

        model_fn = UNet.get_unet
        return model_fn

    if model_name == 'VGG':

        model_fn = VGG.get_vgg
        return model_fn

    if model_name == 'ResNet':

        model_fn = ResNet.get_resnet
        return model_fn

    if model_name == 'VNet':

        model_fn = VNet.get_vnet
        return model_fn


