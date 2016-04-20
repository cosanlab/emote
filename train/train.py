import json
import util.constants as ks
from face_express import FENottinghamCNN, FEBasicCNN
from data.repositories.ck import CKRepository

def setupModel(config_file):
    dict = json.load(config_file)
    train_info = dict[ks.kTrainKey]
    fac_info = train_info[ks.kTrainFACsKey]
    model_name = train_info[ks.kTrainModelKey]
    data_name = train_info[ks.kTrainDataKey]

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
        model = FENottinghamCNN.FENottinghamCNN(repo, fac_info[ks.kTrainFACsCodesKey], fac_info[ks.kTrainFACsIntensityKey])
    elif model_name == ks.kBasicCNN:
        model = FEBasicCNN.FEBasicCNN(repo, fac_info[ks.kTrainFACsCodesKey], fac_info[ks.kTrainFACsIntensityKey])
    else:
        raise RuntimeError("Unable to find a model corresponding to " + model_name)


    return model