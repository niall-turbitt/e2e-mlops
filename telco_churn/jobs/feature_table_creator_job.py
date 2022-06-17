import os

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

        feature_store_params = {'database_name': os.getenv('feature_store_database_name'),
                                'table_name': os.getenv('feature_store_table_name'),
                                'primary_keys': os.getenv('feature_store_table_primary_keys'),
                                'description': os.getenv('feature_store_table_description')}
        labels_table_params = {'database_name': os.getenv('labels_table_database_name'),
                               'table_name': os.getenv('labels_table_name'),
                               'label_col': os.getenv('labels_table_label_col'),
                               'dbfs_path': os.getenv('labels_table_dbfs_path')}

        FeatureTableCreator(data_ingest_params=self.conf['data_ingest_params'],
                            data_prep_params=self.conf['data_prep_params'],
                            feature_store_params=feature_store_params,
                            labels_table_params=labels_table_params).run()

        _logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreatorJob()
    job.launch()
