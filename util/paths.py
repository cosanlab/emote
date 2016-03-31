import os

def get_real_path(file):
    return os.path.dirname(os.path.realpath(file))