import os
import json
import constants as ks

def get_real_path(file_name):
    return os.path.dirname(os.path.realpath(file_name))

def get_project_home():
    return os.path.normpath(get_real_path(__file__) + '/../')

def get_saved_model_path(model_name):
    return '%s/data/saved_models/%s' % (get_project_home(), model_name)

def get_config_path():
    return get_project_home() + '/config.json'


class DataLoc(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataLoc, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.path_map = {}
        config_file = open(get_config_path())
        config_dict = json.load(config_file)
        paths = config_dict[ks.kDataKey]

        for key, val in paths.items() :
            location = val[ks.kDataLocation]

            if(os.path.isabs(location)):
                self.path_map[key] = val[ks.kDataLocation]
            else:
                self.path_map[key] = os.path.normpath(get_real_path(__file__) + "/../" + location)

    def get_path(self, data):
        if self.path_map.has_key(data):
            return self.path_map[data]
        else:
            return None

