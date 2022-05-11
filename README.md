# e2e-mlops

This repo is intended to demonstrate an end-to-end MLOps workflow on Databricks, where a model is deployed along with its ancillary pipelines to a specified (currently single) Databricks workspace.
Each pipeline (e.g model training pipeline, model deployment pipeline) is deployed as a [Databricks job](https://docs.databricks.com/data-engineering/jobs/jobs.html), where these jobs are deployed to a Databricks workspace using Databricks Labs' [`dbx`](https://dbx.readthedocs.io/en/latest/index.html) tool. 

The use case at hand is a churn prediction problem. We use the [IBM Telco Customer Churn dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) to build a simple classifier to predict whether a customer will churn from a fictional telco company.

## Pipelines

The following pipelines currently defined within the package are:
- `demo-setup`
    - Deletes existing feature store tables, and models register to MLflow, in order to start afresh for a demo  
- `feature-table-creation`
    - Creates new feature table and separate labels Delta table
- `model-train`
    - Trains a scikit-learn Random Forest Model  
- `model-deployment`
    - Compare the Staging versus Production models in the MLflow Model Registry. Transition Staging model to Production if outperforming the current Production model
- `model-inference-batch`
    - Load a model from MLflow Model Registry, load features from Feature Store and score batch

---
## Demo
The following outlines the workflow to demo the e2e-mlops repo.

### Set up
1. Spin up an interactive cluster 
1. Clone https://github.com/niall-turbitt/e2e-mlops
1. Configure [Databricks CLI connection profile](https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles)
    - The project is currently configured to use a connection profile called “e2-demo-field-eng”.
    - This is set in [`e2e-mlops/.dbx/project.json`](https://github.com/niall-turbitt/e2e-mlops/blob/main/.dbx/project.json) and configured when the project was originally created with the [`dbx` basic python template](https://dbx.readthedocs.io/en/latest/templates/python_basic.html).
    - If your Databricks CLI connection profile is named something other than “e2-demo-field-eng”, you will need to update the profile field in [`project.json`](https://github.com/niall-turbitt/e2e-mlops/blob/main/.dbx/project.json).
1. Configure Databricks secrets GitHub (for GitHub Actions)
    - **TODO**: link to steps to set this up

    #### ASIDE: Starting from scratch
    
    The following resources should not be present if starting from scratch: 
    - Feature table must be deleted
        - The table `e2e_mlops_testing.churn_features` will be created when `feature-table-creation` pipeline is triggered
        - Currently, the table must be deleted through the Feature Store UI
    - MLflow experiment
        - Either create an experiment via the UI and specify the experiment_id in the model_train.yml conf file or;
        - Specify a path within the workspace to use via the experiment_path param in the model_train.yml conf file
    - Model Registry
        - Delete Model in Model Registry if exists
    
    **NOTE:** As part of the `initial-model-train-register` multitask job, the first task, `demo-setup` will delete these, as specified in [`demo_setup.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/demo_setup.yml).

### Workflow

1. **Run `initial-model-train-register` multitask job**

    - Automated job cluster (**NOTE**: multitask jobs can only be run via `dbx deploy` currently)
        - ``dbx deploy --jobs=initial-model-train-register --files-only``
        - ``dbx launch --job=initial-model-train-register --as-run-submit --trace``
    
    - `initial-model-train-register` tasks:
    1. Demo setup task steps (`demo-setup`)
        1. Delete Model Registry model if exists (archive any existing models)
        1. Delete MLflow experiment if exists
        1. Delete Feature Table if exists
    1. Feature table creation task steps (`feature-table-creation`)
        1. Creates new churn_features feature table in the Feature Store
    1. Model train task steps (`model-train`)
        1. Train initial “baseline” classifier (RandomForestClassifier - `max_depth=4`) 
            - [no changes to config need to be made at this point]
        1. Register the model. Model version 1 will be registered to stage=None upon successful model training.
        1. **Manual Step**: MLflow Model Registry UI promotion to stage='Production'
            - Go to MLflow Model Registry and manually promote model to stage=Production


2. **Code change / model update (Continuous Integration)**

    1. Create new “dev/new_model” branch 
    1. `git checkout -b  dev/new_model`
    1. Make a change to the [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file, updating `max_depth` under model_params from 4 to 8
        - Optional: change run name under mlflow params in [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file
    1. Create pull request, to merge the branch dev/new_model into main

* On pull request the following steps are triggered in the GitHub Actions workflow:
    1. Trigger unit tests 
    1. Trigger pre_model_train_integration tests
    1. Deploy `model-train` job
    1. Deploy `model-deployment` job
    1. Deploy `model-inference-batch` job

3. **Run `model-train` job**
    - Interactive cluster (preferred for demo purposes)
        - ``dbx execute --cluster-name=<name of interactive cluster> --job=model-train``
    - Automated Job Cluster
        - ``dbx deploy --jobs=model-train --files-only``
        - ``dbx launch --job=model-train --as-run-submit --trace``

    - Model train job steps (`model-train`)
        1. Train improved “new” classifier (RandomForestClassifier - `max_depth=8`)
        1. Register the model. Model version 2 will be registered to stage=None upon successful model training.
        1. **Manual Step**: MLflow Model Registry UI promotion to stage='Staging'
            - Go to Model registry and manually promote model to stage='Staging'

    **ASIDE:** At this point, there should now be two model versions register in MLflow Model Registry:
        
    - Version 1 (Production): RandomForestClassifier (`max_depth=4`)
    - Version 2 (Staging): RandomForestClassifier (`max_depth=8`)


4. **Run `model-deployment` job (Continuous Deployment)**
    - Interactive cluster (preferred for demo purposes)
        - ``dbx execute --cluster-name=<name of interactive cluster> --job=model-deployment``
    - Automated Job Cluster
        - ``dbx deploy --jobs=model-deployment --files-only``
        - ``dbx launch --job=model-deployment --as-run-submit --trace``
    
    - Model deployment job steps  (`model-deployment`)
        1. Compare new “candidate model” in stage='Staging' versus current Production model in stage='Production'
        1. Comparison criteria set through [`model_deployment.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_deployment.yml)
            1. Compute predictions using both models against a specified reference dataset
            1. If Staging model performs better than Production model, promote Staging model to Production and archive existing Production model
            1. If Staging model performs worse than Production model, archive Staging model
            

5. **Run `model-inference-batch` job** 

    - Interactive cluster (preferred for demo purposes)
        - ``dbx execute --cluster-name=<name of interactive cluster> --job=model-inference-batch``
    - Automated Job Cluster
        - ``dbx deploy --jobs=model-inference-batch --files-only``
        - ``dbx launch --job=model-inference-batch --as-run-submit --trace``

    - Batch model inference steps  (`model-inference`)
        1. Load model from stage=Production in Model Registry
            - **NOTE:** model must have been logged to MLflow using the Feature Store API
        1. Use primary keys in specified inference input data to load features from feature store
        1. Apply loaded model to loaded features
        1. Write predictions to specified Delta path




---
## Development

While using this project, you need Python 3.X and `pip` or `conda` for package management.

### Installing project requirements

```bash
pip install -r unit-requirements.txt
```

### Install project package in a developer mode

```bash
pip install -e .
```

### Testing

For local unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

For an integration test on interactive cluster, use the following command:
```
dbx execute --cluster-name=<name of interactive cluster> --job=e2e-mlops-sample-integration-test
```

For a test on an automated job cluster, deploy the job files and then launch:
```
dbx deploy --jobs=e2e-mlops-sample-integration-test --files-only
dbx launch --job=e2e-mlops-sample-integration-test --as-run-submit --trace
```

### Interactive execution and development

1. `dbx` expects that cluster for interactive execution supports `%pip` and `%conda` magic [commands](https://docs.databricks.com/libraries/notebooks-python-libraries.html).
2. Please configure your job in `conf/deployment.yml` file.
2. To execute the code interactively, provide either `--cluster-id` or `--cluster-name`.
```bash
dbx execute \
    --cluster-name="<some-cluster-name>" \
    --job=job-name
```

Multiple users also can use the same cluster for development. Libraries will be isolated per each execution context.

### Preparing deployment file

Next step would be to configure your deployment objects. To make this process easy and flexible, we're using YAML for configuration.

By default, deployment configuration is stored in `conf/deployment.yml`.

### Deployment for Run Submit API

To deploy only the files and not to override the job definitions, do the following:

```bash
dbx deploy --files-only
```

To launch the file-based deployment:
```
dbx launch --as-run-submit --trace
```

This type of deployment is handy for working in different branches, not to affect the main job definition.

### Deployment for Run Now API

To deploy files and update the job definitions:

```bash
dbx deploy
```

To launch the file-based deployment:
```
dbx launch --job=<job-name>
```

This type of deployment shall be mainly used from the CI pipeline in automated way during new release.


### CICD pipeline settings

Please set the following secrets or environment variables for your CI provider:
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

### Testing and releasing via CI pipeline

- To trigger the CI pipeline, simply push your code to the repository. If CI provider is correctly set, it shall trigger the general testing pipeline
- To trigger the release pipeline, get the current version from the `telco_churn/__init__.py` file and tag the current code version:
```
git tag -a v<your-project-version> -m "Release tag for version <your-project-version>"
git push origin --tags
```
