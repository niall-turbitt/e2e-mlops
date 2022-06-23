import logging


def get_logger():
    logging.getLogger('py4j.java_gateway').setLevel(logging.ERROR)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
    return logging.getLogger(__name__)
