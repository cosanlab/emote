import util.constants as ks
from util import config
import FENottinghamCNN, FEBasicCNN
from data.repositories.ck import CKRepository

def modelForTraining():
    model_info = config.get_model_info()
    fac_info = model_info[ks.kModelFACsKey]
    model_name = model_info[ks.kModelNameKey]
    data_name = model_info[ks.kModelDataKey]

    data_info = config.get_data_info(data_name)

    #Get datset interface
    ##All will be an instance of FACRepository

    repo = None
    if data_name == ks.kDataCK:
        repo = CKRepository()
    else:
        raise RuntimeError("Unable to find a dataset corresponding to " + data_name)

    #Get the correct model
    model = None


    if model_name == ks.kNottinghamCNN:
        model = FENottinghamCNN.FENottinghamCNN(repo, fac_info[ks.kModelFACsCodesKey], fac_info[ks.kModelFACsIntensityKey])
    elif model_name == ks.kBasicCNN:
        model = FEBasicCNN.FEBasicCNN(model_info[ks.kDataImageSize], fac_info[ks.kModelFACsCodesKey], repo, fac_info[ks.kModelFACsIntensityKey])
    else:
        raise RuntimeError("Unable to find a model corresponding to " + model_name)


    return model