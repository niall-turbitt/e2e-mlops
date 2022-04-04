from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class PipelineCreator:

    @classmethod
    def make_baseline(cls, model_params: dict) -> Pipeline:

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric_transformer",
                 SimpleImputer(strategy="median"),
                 make_column_selector(dtype_exclude='object')
                 ),
                ("categorical_transformer",
                 OneHotEncoder(handle_unknown='ignore'),
                 make_column_selector(dtype_include='object')
                 ),
            ],
            remainder="passthrough",
            sparse_threshold=0
        )

        standardizer = StandardScaler()
        xgb_classifier = XGBClassifier(use_label_encoder=False,
                                       **model_params)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("standardizer", standardizer),
            ("classifier", xgb_classifier),
        ])

        return pipeline
