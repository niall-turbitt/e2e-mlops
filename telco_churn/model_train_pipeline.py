from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


class ModelTrainPipeline:

    @classmethod
    def create_train_pipeline(cls, model_params: dict) -> Pipeline:

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric_transformer',
                 SimpleImputer(strategy='median'),
                 make_column_selector(dtype_exclude='object')
                 ),
                ('categorical_transformer',
                 OneHotEncoder(handle_unknown='ignore'),
                 make_column_selector(dtype_include='object')
                 ),
            ],
            remainder='passthrough',
            sparse_threshold=0
        )

        rf_classifier = RandomForestClassifier(**model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', rf_classifier),
        ])

        return pipeline
