import os
import json
import constants as ks

def get_real_path(file_name):
    """Gets the director of the file
    
    :param file_name: File to get dir or
    :type file_name: str
    :returns: str - path of containing directory
    """
    return os.path.dirname(os.path.realpath(file_name))

def get_project_home():
    """Gets root directory of the project

    :returns: str
    """
    return os.path.normpath(get_real_path(__file__) + '/../../')

def get_source_home():
    """Gets path to the src folder

    :returns: str
    """
    return os.path.normpath(get_real_path(__file__) + '/../')

def get_saved_model_path(model_name):
    """Gets path of most recently saved model given its name
    
    :param model_name: Constant name of model
    :type model_name: str

    :returns: str - path to saved model
    """
    return '%s/data/saved_models/%s' % (get_project_home(), model_name)

def get_config_path():
    """Gets expected path to config.json

    :returns: str 
    """
    return get_project_home() + '/config.json'


class DataLoc(object):
    """Handles getting and absoluting the locations of data sets 
    """

    _instance = None
    def __new__(cls, *args, **kwargs):
        """ Make this a singleton object, is it saves itself
        """
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
        """Given the name of a dataset (as defined in util.constants), returns absolute path to data set
        
        :param data: name of data set in constants.py
        :type data: str
        :returns: str - path to data set
        """
        if self.path_map.has_key(data):
            return self.path_map[data]
        else:
            return None

