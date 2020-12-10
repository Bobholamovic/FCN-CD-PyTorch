# Custom model builders

from core.misc import MODELS


@MODELS.register_func('Unet_model')
def build_Unet_model(C):
    from models.unet import Unet
    return Unet(6, 2)


@MODELS.register_func('Unet_OSCD_model')
def build_Unet_OSCD_model(C):
    from models.unet import Unet
    return Unet(26, 2)


@MODELS.register_func('SiamUnet_diff_model')
def build_SiamUnet_diff_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(3, 2)


@MODELS.register_func('SiamUnet_diff_OSCD_model')
def build_SiamUnet_diff_OSCD_model(C):
    from models.siamunet_diff import SiamUnet_diff
    return SiamUnet_diff(13, 2)


@MODELS.register_func('SiamUnet_conc_model')
def build_SiamUnet_conc_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(3, 2)


@MODELS.register_func('SiamUnet_conc_OSCD_model')
def build_SiamUnet_conc_OSCD_model(C):
    from models.siamunet_conc import SiamUnet_conc
    return SiamUnet_conc(13, 2)


@MODELS.register_func('FresUnet_model')
def build_FresUnet_model(C):
    from models.fresunet import FresUNet
    return FresUNet(6, 2)


@MODELS.register_func('FresUnet_OSCD_model')
def build_FresUnet_OSCD_model(C):
    from models.fresunet import FresUNet
    return FresUNet(26, 2)