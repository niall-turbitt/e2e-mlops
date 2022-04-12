import pandas as pd
from sklearn.metrics import roc_auc_score


class ModelEvaluation:

    def __init__(self, metric_prefix: str = "test_"):
        self.metric_prefix = metric_prefix

    @staticmethod
    def _roc_auc_score(y_true: pd.Series, y_score: pd.Series):
        return roc_auc_score(y_true=y_true,
                             y_score=y_score,
                             average="weighted",
                             multi_class="ovo")

    def evaluate(self, y_true: pd.Series, y_score: pd.Series) -> dict:
        return {
            f"{self.metric_prefix}roc_auc": self._roc_auc_score(y_true, y_score),
        }
