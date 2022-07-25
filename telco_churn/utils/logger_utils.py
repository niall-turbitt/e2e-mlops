import logging


class NoReceivedCommandFilter(logging.Filter):
    def filter(self, record):
        if 'Received command c' not in record.getMessage():
            return record.getMessage()


class NoPythonDotEnvFilter(logging.Filter):
    def filter(self, record):
        if 'Python-dotenv' not in record.getMessage():
            return record.getMessage()


def get_logger():
    logging.getLogger('py4j.java_gateway').setLevel(logging.ERROR)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    filter_1 = NoReceivedCommandFilter()
    filter_2 = NoPythonDotEnvFilter()
    logger.addFilter(filter_1)
    logger.addFilter(filter_2)

    return logger
