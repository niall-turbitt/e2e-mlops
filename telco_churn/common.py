"""
Module containing common data classes used throughout different pipelines, in addition to Workload class which is
extended to run pipelines/tasks.
"""
import os
import sys
from dataclasses import dataclass

import yaml
import pathlib
import dotenv
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from logging import Logger
from typing import Dict, Any, Union, List
from pyspark.sql import SparkSession


@dataclass
class MLflowTrackingConfig:
    """
    Configuration data class used to unpack MLflow parameters during a model training run.

    Attributes:
        run_name (str)
            Name of MLflow run
        experiment_id (int)
            ID of the MLflow experiment to be activated. If an experiment with this ID does not exist, raise an exception.
        experiment_path (str)
            Case sensitive name of the experiment to be activated. If an experiment with this name does not exist,
            a new experiment wth this name is created.
        model_name (str)
            Name of the registered model under which to create a new model version. If a registered model with the given
            name does not exist, it will be created automatically.
    """
    run_name: str
    experiment_id: int = None
    experiment_path: str = None
    model_name: str = None


@dataclass
class FeatureStoreTableConfig:
    """
    Configuration data class used to unpack parameters when creating or loading a Feature Store table.

    Attributes:
        database_name (str)
            Name of database to use for creating the feature table
        table_name (str)
            Name of feature table
        primary_keys (string or list)
            String or list of strings, of columns to use as the primary key(s). Use single column (customerID) as the
            primary key for the telco churn example.
        description (str)
            [Optional] string containing description to attribute to the feature table in the Feature Store.
            Only used when creating a Feature Store table.
    """
    database_name: str
    table_name: str
    primary_keys: Union[str, List[str]]
    description: str = None


@dataclass
class LabelsTableConfig:
    """
    Configuration data class used to unpack parameters when creating or loading labels table.

    Attributes:
        database_name (str)
            Name of database to use for creating the labels table
        table_name (str)
            Name of labels table within the database
        label_col (str)
            Name of column to use as the label column (in telco churn example we rename this column to 'churn')
        primary_keys (string or list)
            [Optional] String or list of strings, of columns to use as the primary key(s)
        dbfs_path (str)
            [Optional] DBFS path to use for the labels table (saving as a Delta table)
    """
    database_name: str
    table_name: str
    label_col: str
    primary_keys: Union[str, List[str]] = None
    dbfs_path: str = None


class Workload(ABC):
    """
    This is an abstract class that provides handy interfaces to implement workloads (e.g. pipelines or job tasks).
    Create a child from this class and implement the abstract launch method.
    Class provides access to the following useful objects:
    * self.spark is a SparkSession
    * self.dbutils provides access to the DBUtils
    * self.logger provides access to the Spark-compatible logger
    * self.conf provides access to the parsed configuration of the job
    * self.env_vars provides access to the parsed environment variables of the job
    """
    def __init__(self, spark=None, init_conf=None):
        self.spark = self._prepare_spark(spark)
        self.logger = self._prepare_logger()
        self.dbutils = self.get_dbutils()
        if init_conf:
            self.conf = init_conf
        else:
            self.conf = self._provide_config()
        self._log_conf()
        self.env_vars = self.get_env_vars_as_dict()
        self._log_env_vars()

    @staticmethod
    def _prepare_spark(spark) -> SparkSession:
        if not spark:
            return SparkSession.builder.getOrCreate()
        else:
            return spark

    @staticmethod
    def _get_dbutils(spark: SparkSession):
        try:
            from pyspark.dbutils import DBUtils  # noqa

            if 'dbutils' not in locals():
                utils = DBUtils(spark)
                return utils
            else:
                return locals().get('dbutils')
        except ImportError:
            return None

    def get_dbutils(self):
        utils = self._get_dbutils(self.spark)

        if not utils:
            self.logger.warn('No DBUtils defined in the runtime')
        else:
            self.logger.info('DBUtils class initialized')

        return utils

    def _provide_config(self):
        self.logger.info('Reading configuration from --conf-file job option')
        conf_file = self._get_conf_file()
        if not conf_file:
            self.logger.info(
                'No conf file was provided, setting configuration to empty dict.'
                'Please override configuration in subclass init method'
            )
            return {}
        else:
            self.logger.info(f'Conf file was provided, reading configuration from {conf_file}')
            return self._read_config(conf_file)

    @staticmethod
    def _get_conf_file():
        p = ArgumentParser()
        p.add_argument('--conf-file', required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.conf_file

    @staticmethod
    def _read_config(conf_file) -> Dict[str, Any]:
        config = yaml.safe_load(pathlib.Path(conf_file).read_text())
        return config

    @staticmethod
    def _get_base_data_params():
        p = ArgumentParser()
        p.add_argument('--base-data-params', required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.base_data_params

    @staticmethod
    def _get_env():
        p = ArgumentParser()
        p.add_argument('--env', required=False, type=str)
        namespace = p.parse_known_args(sys.argv[1:])[0]
        return namespace.env

    @staticmethod
    def _set_environ(env_vars):
        dotenv.load_dotenv(env_vars)

    def get_env_vars_as_dict(self):
        base_data_params = self._get_base_data_params()
        self._set_environ(base_data_params)

        env = self._get_env()
        self._set_environ(env)

        return dict(os.environ)

    def _prepare_logger(self) -> Logger:
        log4j_logger = self.spark._jvm.org.apache.log4j  # noqa
        return log4j_logger.LogManager.getLogger(self.__class__.__name__)

    def _log_conf(self):
        # log parameters
        self.logger.info('Launching job with configuration parameters:')
        for key, item in self.conf.items():
            self.logger.info('\t Parameter: %-30s with value => %-30s' % (key, item))

    def _log_env_vars(self):
        # log parameters
        self.logger.info('Using environment variables:')
        for key, item in self.env_vars.items():
            self.logger.info('\t Parameter: %-30s with value => %-30s' % (key, item))

    @abstractmethod
    def launch(self):
        """
        Main method of the job.
        :return:
        """
        pass


def get_dbutils(
    spark: SparkSession,
):  # please note that this function is used in mocking by its name
    try:
        from pyspark.dbutils import DBUtils  # noqa

        if "dbutils" not in locals():
            utils = DBUtils(spark)
            return utils
        else:
            return locals().get("dbutils")
    except ImportError:
        return None
