import paths
import json
import util.constants as ks

def get_config_dict():
    fp = get_config_pointer()
    diction = json.load(fp)
    fp.close()
    return diction

def get_config_pointer():
    fp = open(paths.get_config_path())
    if fp is None:
        print("Error: Unable to find config.json file in project home")

    return fp

def get_detector_name():
    return get_config_dict()[ks.kDetectorKey]

def get_data_info(data_name):
    config = get_config_dict()
    data = config[ks.kDataKey]

    if data_name == ks.kDataCK:
        return data[ks.kDataCK]
    elif data_name == ks.kDataAMFED:
        return data[ks.kDataAMFED]
    elif data_name == ks.kDataDIFSA:
        return data[ks.kDataDIFSA]
    else:
        print("Error: could not find required data: %s" % data_name)
        return None

def get_model_info():
    config = get_config_dict()
    model = config['model']

    return model
