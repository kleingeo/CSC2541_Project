import logging.config
import os.path
# import logging


path = os.path.dirname(__file__)


LOGGING_CONF = os.path.join(path, 'logging_configuration.ini')
# logging.basicConfig(filename='change.log')
logging.config.fileConfig(LOGGING_CONF)

