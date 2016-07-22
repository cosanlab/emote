import util.constants as ks
from util import config
import face_express.FEAggregatedCNN
import face_express.FEMultiLabelCNN
from face_express import FEGhoshCNN
from data.repositories import CKRepository, DIFSARepository

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
    elif data_name == ks.kDataDIFSA:
        repo = DIFSARepository()
    else:
        raise RuntimeError("Unable to find a dataset corresponding to " + data_name)

    #Get the correct model
    model = None

    if model_name == ks.kAggregatedCNN:
        model = FEAggregatedCNN(model_info[ks.kDataImageSize], fac_info[ks.kModelFACsCodesKey], repo)
    elif model_name == ks.kMultiLabelCNN:
        model = FEMultiLabelCNN(model_info[ks.kDataImageSize], fac_info[ks.kModelFACsCodesKey], repo)
    elif model_name == ks.kGhoshCNN:
        model = FEGhoshCNN(model_info[ks.kDataImageSize], fac_info[ks.kModelFACsCodesKey], repo)
    else:
        raise RuntimeError("Unable to find a model corresponding to " + model_name)


    return model
