from telco_churn.common import Workload, MLflowTrackingConfig, FeatureStoreTableConfig, LabelsTableConfig
from telco_churn.model_train import ModelTrain, ModelTrainConfig
from telco_churn.utils.logger_utils import get_logger

_logger = get_logger()


class ModelTrainJob(Workload):

    def _get_mlflow_tracking_cfg(self):
        try:
            experiment_id = self.env_vars['model_train_experiment_id']
        except KeyError:
            experiment_id = None
        try:
            experiment_path = self.env_vars['model_train_experiment_path']
        except KeyError:
            experiment_path = None

        return MLflowTrackingConfig(run_name=self.conf['mlflow_params']['run_name'],
                                    experiment_id=experiment_id,
                                    experiment_path=experiment_path,
                                    model_name=self.env_vars['model_name'])

    def _get_feature_store_table_cfg(self):
        return FeatureStoreTableConfig(database_name=self.env_vars['feature_store_database_name'],
                                       table_name=self.env_vars['feature_store_table_name'],
                                       primary_keys=self.env_vars['feature_store_table_primary_keys'])

    def _get_labels_table_cfg(self):
        return LabelsTableConfig(database_name=self.env_vars['labels_table_database_name'],
                                 table_name=self.env_vars['labels_table_name'],
                                 label_col=self.env_vars['labels_table_label_col'])

    def _get_pipeline_params(self):
        return self.conf['pipeline_params']

    def _get_model_params(self):
        return self.conf['model_params']

    def launch(self):
        _logger.info('Launching ModelTrainJob job')
        _logger.info(f'Running model-train pipeline in {self.env_vars["env"]} environment')
        cfg = ModelTrainConfig(mlflow_tracking_cfg=self._get_mlflow_tracking_cfg(),
                               feature_store_table_cfg=self._get_feature_store_table_cfg(),
                               labels_table_cfg=self._get_labels_table_cfg(),
                               pipeline_params=self._get_pipeline_params(),
                               model_params=self._get_model_params(),
                               conf=self.conf,
                               env_vars=self.env_vars)
        ModelTrain(cfg).run()
        _logger.info('ModelTrainJob job finished!')


if __name__ == '__main__':
    job = ModelTrainJob()
    job.launch()
