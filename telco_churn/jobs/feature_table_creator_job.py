from telco_churn.common import Job
from telco_churn.feature_table_creator import FeatureTableCreator
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class FeatureTableCreatorJob(Job):

    def launch(self) -> None:
        """
        Launch FeatureStoreTableCreator job
        """
        _logger.info("Launching FeatureTableCreator job")
        FeatureTableCreator(**self.conf).run()
        _logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreatorJob()
    job.launch()
