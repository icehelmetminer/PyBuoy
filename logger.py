import logging
import os

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    file_handler = logging.FileHandler('logs/pybuoy.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
