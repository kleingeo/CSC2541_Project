import OtherModels.IUNet as IUNet
import OtherModels.UNet as UNet
import OtherModels.VGG as VGG
import OtherModels.ResNet as ResNet
import OtherModels.VNet as VNet

def ModelSelectUtil(model_name):


    assert model_name in ['IUNet', 'UNet', 'VGG', 'ResNet', 'VNet'], \
        'Must select either appropriate model architecture. Must be "IUNet", "UNet", "VNet", "VGG", or "ResNet".'

    if model_name == 'IUNet':

        model = IUNet.get_iunet

    if model_name == 'UNet':

        model = UNet.get_unet

    if model_name == 'VGG':

        model = VGG.get_vgg

    if model_name == 'ResNet':

        model_fn = ResNet.get_resnet

    if model_name == 'VNet':

        model_fn = VNet.get_vnet

