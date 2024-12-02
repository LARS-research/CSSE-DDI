import json
import logging
import logging.config
from os import makedirs

def get_logger(log_dir, name):
    config_dict = json.load(open('./config/' + 'logger_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name + '.log'
    makedirs(log_dir, exist_ok=True)
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger