import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from telco_churn.model_train_pipeline import ModelTrainPipeline


class ModelTrainPipelineTest(unittest.TestCase):

    def test_create_train_pipeline(self):
        @dataclass
        class Example:
            contract: str
            dependents: str
            deviceProtection: str
            gender: str
            internetService: str
            monthlyCharges: float
            multipleLines: str
            onlineBackup: str
            onlineSecurity: str
            paperlessBilling: str
            partner: str
            paymentMethod: str
            phoneService: str
            seniorCitizen: float
            streamingMovies: str
            streamingTV: str
            techSupport: str
            tenure: float
            totalCharges: float

        X = pd.DataFrame(data=[
            Example('Two year', 'Yes', 'No', 'Female', 'DSL', 53.65, 'No phone service', 'Yes', 'No', 'No', 'Yes',
                    'Credit card (automatic)', 'No', 0.0, 'Yes', 'Yes', 'Yes', 72.0, 3784.0),
            Example('Month-to-month', 'No', 'No', 'Male', 'Fiber optic', 74.9, 'Yes', 'No', 'No', 'Yes', 'No',
                    'Electronic check', 'Yes', 0.0, 'No', 'No', 'No', 1.0, 74.9),
            Example('Month-to-month', 'No', 'No', 'Female', 'Fiber optic', 100.4, 'Yes', 'No', 'No', 'Yes', 'Yes',
                    'Bank transfer (automatic)', 'Yes', 1.0, 'Yes', 'Yes', 'Yes', 58.0, 5749.8),
        ])
        y = np.random.randint(2, size=3)

        model_params = {'n_estimators': 4,
                        'max_depth': 4,
                        'min_samples_leaf': 1,
                        'max_features': 'auto',
                        'random_state': 42}

        pipeline = ModelTrainPipeline.create_train_pipeline(model_params=model_params)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert np.array_equal(y_pred, y_pred.astype(bool))

