from .InceptionResnetV2Pytorch import InceptionResnetV2Pytorch
from .regnet import Regnet
MODELS = dict(inception_resnetv2_pytorch=InceptionResnetV2Pytorch,
              regnet=Regnet
              # other models
              )


def load_model(model_name, **kwargs):
    return MODELS[model_name](**kwargs).get_model()
