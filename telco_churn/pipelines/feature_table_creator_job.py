from telco_churn.common import Workload, FeatureStoreTableConfig, LabelsTableConfig
from telco_churn.feature_table_creator import FeatureTableCreator, FeatureTableCreatorConfig
from telco_churn.featurize import FeaturizerConfig
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class FeatureTableCreatorJob(Workload):

    def _get_input_table(self) -> dict:
        return self.conf['input_table']

    def _get_data_prep_params(self) -> FeaturizerConfig:
        return FeaturizerConfig(**self.conf['data_prep_params'])

    def _get_feature_store_table_cfg(self) -> FeatureStoreTableConfig:
        return FeatureStoreTableConfig(database_name=self.env_vars['feature_store_database_name'],
                                       table_name=self.env_vars['feature_store_table_name'],
                                       primary_keys=self.env_vars['feature_store_table_primary_keys'],
                                       description=self.env_vars['feature_store_table_description'])

    def _get_labels_table_cfg(self) -> LabelsTableConfig:
        return LabelsTableConfig(database_name=self.env_vars['labels_table_database_name'],
                                 table_name=self.env_vars['labels_table_name'],
                                 label_col=self.env_vars['labels_table_label_col'],
                                 dbfs_path=self.env_vars['labels_table_dbfs_path'])

    def launch(self) -> None:
        """
        Launch FeatureStoreTableCreator job
        """
        _logger.info('Launching FeatureTableCreator job')
        _logger.info(f'Running feature-table-creation pipeline in {self.env_vars["env"]} environment')
        cfg = FeatureTableCreatorConfig(input_table=self._get_input_table(),
                                        featurizer_cfg=self._get_data_prep_params(),
                                        feature_store_table_cfg=self._get_feature_store_table_cfg(),
                                        labels_table_cfg=self._get_labels_table_cfg())
        FeatureTableCreator(cfg).run()
        _logger.info('FeatureTableCreator job finished!')


if __name__ == '__main__':
    job = FeatureTableCreatorJob()
    job.launch()
