import os
import json
import constants as ks

def get_real_path(file):
    return os.path.dirname(os.path.realpath(file))

def get_project_home():
    return os.path.normpath(get_real_path(__file__) + '/../')

class DataLoc(object):
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataLoc, cls).__new__(
                                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.path_map = {}
        f = open(get_real_path(__file__) + '/../config.json')
        parsed = json.load(f)
        paths = parsed[ks.kDataKey]

        for key, val in paths.items() :
            if(os.path.isabs(val)):
                self.path_map[key] = val
            else:
                self.path_map[key] = os.path.normpath(get_real_path(__file__) + "/../" + val)

    def get_path(self, data):
        if self.path_map.has_key(data):
            return self.path_map[data]
        else:
            return None

