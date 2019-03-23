
import logging
from logging.config import dictConfig
import sys


def logging_dict_config(filename=None):

    if filename is None:
        filename = 'log.log'

    logging_config = dict(

        version=1,
        formatters={
            'f': {'format':
                  '%(asctime)s.%(msecs)03d %(levelname)s : %(filename)s : Line %(lineno)s : %(funcName)20s() : %(message)s',
                  'datefmt': '%Y-%m-%d %H:%M:%S',
                  'class': 'logging.Formatter'}
            },
        handlers={
            'console': {'class': 'logging.StreamHandler',
                        'formatter': 'f',
                        'level': 'NOTSET',
                        'stream': sys.stdout},

            'error_file': {'class': 'logging.handlers.RotatingFileHandler',
                           'formatter': 'f',
                           'level': 'NOTSET',
                           'filename': filename,
                           'maxBytes': 100000000,
                           'backupCount': 5},
            },
        root={
            'handlers': ['console', 'error_file'],
            'level': 'NOTSET',
            'formatter': 'f'
            },
    )

    return logging_config

