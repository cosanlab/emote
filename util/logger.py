import logging
import datetime
import os

from paths import get_project_home

def setupLogging(logLevel=logging.DEBUG):
    logging.root.setLevel(logLevel)

    log_path = get_project_home() + "/logs/emote-" + str(datetime.datetime.now()) + '.log'
    os.system('touch "' + log_path + '"')
    logging.basicConfig(filename=log_path, level=logLevel, format="%(asctime)s - emote: %(name)s(%(process)d) - %(levelname)s - %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')

