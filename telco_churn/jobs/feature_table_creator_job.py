import os

from telco_churn.common import Job
from telco_churn.feature_table_creator import FeatureTableCreator
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class FeatureTableCreatorJob(Job):

    def _get_data_ingest_params(self):
        return self.conf['data_ingest_params']

    def _get_data_prep_params(self):
        return self.conf['data_prep_params']

    @staticmethod
    def _get_feature_store_params():
        return {'database_name': os.getenv('feature_store_database_name'),
                'table_name': os.getenv('feature_store_table_name'),
                'primary_keys': os.getenv('feature_store_table_primary_keys'),
                'description': os.getenv('feature_store_table_description')}

    @staticmethod
    def _get_labels_table_params():
        return {'database_name': os.getenv('labels_table_database_name'),
                'table_name': os.getenv('labels_table_name'),
                'label_col': os.getenv('labels_table_label_col'),
                'dbfs_path': os.getenv('labels_table_dbfs_path')}

    def launch(self) -> None:
        """
        Launch FeatureStoreTableCreator job
        """
        _logger.info("Launching FeatureTableCreator job")
        _logger.info(f'Running feature-table-creation pipeline in {os.getenv("DEPLOYMENT_ENV")} environment')
        FeatureTableCreator(data_ingest_params=self._get_data_ingest_params(),
                            data_prep_params=self._get_data_prep_params(),
                            feature_store_params=self._get_feature_store_params(),
                            labels_table_params=self._get_labels_table_params()).run()
        _logger.info("FeatureTableCreator job finished!")


if __name__ == "__main__":
    job = FeatureTableCreatorJob()
    job.launch()
