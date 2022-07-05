# e2e-mlops

This repo is intended to demonstrate an end-to-end MLOps workflow on Databricks, where a model is deployed along with its ancillary pipelines to a specified (currently single) Databricks workspace.
Each pipeline (e.g model training pipeline, model deployment pipeline) is deployed as a [Databricks job](https://docs.databricks.com/data-engineering/jobs/jobs.html), where these jobs are deployed to a Databricks workspace using Databricks Labs' [`dbx`](https://dbx.readthedocs.io/en/latest/index.html) tool. 

The use case at hand is a churn prediction problem. We use the [IBM Telco Customer Churn dataset](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) to build a simple classifier to predict whether a customer will churn from a fictional telco company.

Note that the package is solely developed via an IDE, and as such there are no Databricks Notebooks in the repository. All jobs are executed via a command line based workflow using [`dbx`](https://dbx.readthedocs.io/en/latest/).

## Pipelines

The following pipelines currently defined within the package are:
- `demo-setup`
    - Deletes existing feature store tables, existing MLflow experiments and models registered to MLflow Model Registry, 
      in order to start afresh for a demo.  
- `feature-table-creation`
    - Creates new feature table and separate labels Delta table.
- `model-train`
    - Trains a scikit-learn Random Forest model  
- `model-deployment`
    - Compare the Staging versus Production models in the MLflow Model Registry. Transition the Staging model to 
      Production if outperforming the current Production model.
- `model-inference-batch`
    - Load a model from MLflow Model Registry, load features from Feature Store and score batch.

## Demo
The following outlines the workflow to demo the repo.

### Set up
1. Clone https://github.com/niall-turbitt/e2e-mlops
   - `git clone https://github.com/niall-turbitt/e2e-mlops.git`
1. Configure [Databricks CLI connection profile](https://docs.databricks.com/dev-tools/cli/index.html#connection-profiles)
    - The project is designed to use 3 different Databricks CLI connection profiles: dev, staging and prod. 
      These profiles are set in [e2e-mlops/.dbx/project.json](https://github.com/niall-turbitt/e2e-mlops/blob/main/.dbx/project.json).
    - Note that for demo purposes we use the same connection profile for each of the 3 environments. 
      **In practice each profile would correspond to separate dev, staging and prod Databricks workspaces.**
    - This [project.json](https://github.com/niall-turbitt/e2e-mlops/blob/main/.dbx/project.json) file will have to be 
      adjusted accordingly to the connection profiles a user has configured on their local machine.
1. Configure Databricks secrets for GitHub Actions
    - Within the GitHub project navigate to Secrets under the project settings
    - To run the GitHub actions workflows we require the following GitHub actions secrets:
        - `DATABRICKS_STAGING_HOST`
            - URL of Databricks staging workspace
        - `DATABRICKS_STAGING_TOKEN`
            - [Databricks access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for staging workspace
        - `DATABRICKS_PROD_HOST`
            - URL of Databricks production workspace
        - `DATABRICKS_PROD_TOKEN`
            - [Databricks access token](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for production workspace
        - `GH_TOKEN`
            - GitHub [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

    #### ASIDE: Starting from scratch
    
    The following resources should not be present if starting from scratch: 
    - Feature table must be deleted
        - The table e2e_mlops_testing.churn_features will be created when the feature-table-creation pipeline is triggered.
    - MLflow experiment
        - MLflow Experiments during model training and model deployment will be used in both the dev and prod environments. 
          The paths to these experiments are configured in [conf/deployment.yml](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/deployment.yml).
        - For demo purposes, we delete these experiments if they exist to begin from a blank slate.
    - Model Registry
        - Delete Model in MLflow Model Registry if exists.
    
    **NOTE:** As part of the `initial-model-train-register` multitask job, the first task `demo-setup` will delete these, 
   as specified in [`demo_setup.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/demo_setup.yml).

### Workflow

1. **Run `telco-churn-initial-model-train-register` multitask job**

    - To demonstrate a CICD workflow, we want to start from a “steady state” where there is a current model in production. 
      As such, we will manually trigger a multitask job to do the following steps:
      1. Set up the workspace for the demo by deleting existing MLflow experiments and register models, along with 
         existing Feature Store and labels tables. 
      1. Create a new Feature Store table to be used by the model training pipeline.
      1. Train an initial “baseline” model
    - There is then a final manual step to promote this newly trained model to production via the MLflow Model Registry UI.

    - Outlined below are the detailed steps to do this:

        1. Run the multitask `telco-churn-initial-model-train-register` job via an automated job cluster 
           (NOTE: multitask jobs can only be run via `dbx deploy; dbx launch` currently).
           ```
           dbx deploy --jobs=telco-churn-initial-model-train-register --environment=prod --files-only
           dbx launch --job=telco-churn-initial-model-train-register --environment=prod --as-run-submit --trace
           ```
           See the Limitations section below regarding running multitask jobs. In order to reduce cluster start up time
           you may want to consider using a [Databricks pool](https://docs.databricks.com/clusters/instance-pools/index.html), 
           and specify this pool ID in [`conf/deployment.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/deployment.yml).
    - `telco-churn-initial-model-train-register` tasks:
        1. Demo setup task steps ([`demo-setup`](https://github.com/niall-turbitt/e2e-mlops/blob/main/telco_churn/jobs/demo_setup_job.py))
            1. Delete Model Registry model if exists (archive any existing models).
            1. Delete MLflow experiment if exists.
            1. Delete Feature Table if exists.
        1. Feature table creation task steps (`feature-table-creation`)
            1. Creates new churn_features feature table in the Feature Store
        1. Model train task steps (`model-train`)
            1. Train initial “baseline” classifier (RandomForestClassifier - `max_depth=4`) 
                - **NOTE:** no changes to config need to be made at this point
            1. Register the model. Model version 1 will be registered to `stage=None` upon successful model training.
            1. **Manual Step**: MLflow Model Registry UI promotion to `stage='Production'`
                - Go to MLflow Model Registry and manually promote model to `stage='Production'`.


2. **Code change / model update (Continuous Integration)**

    - Create new “dev/new_model” branch 
        - `git checkout -b  dev/new_model`
    - Make a change to the [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file, updating `max_depth` under model_params from 4 to 8
        - Optional: change run name under mlflow params in [`model_train.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_train.yml) config file
    - Create pull request, to merge the branch dev/new_model into main

* On pull request the following steps are triggered in the GitHub Actions workflow:
    1. Trigger unit tests 
    1. Trigger integration tests


3. **Cut release**

    - Create tag (e.g. `v0.0.1`)
        - `git tag <tag_name> -a -m “Message”`
            - Note that tags are matched to `v*`, i.e. `v1.0`, `v20.15.10`
    - Push tag
        - `git push origin <tag_name>`

    - On pushing this the following steps are triggered in the [`onrelease.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/.github/workflows/onrelease.yml) GitHub Actions workflow:
        1. Trigger unit tests
        1. Deploy `telco-churn-model-train` job
        1. Deploy `telco-churn-model-deployment` job
        1. Deploy `telco-churn-model-inference-batch` job
            - These jobs will now all be present in the specified workspace, and visible under the [Workflows](https://docs.databricks.com/data-engineering/jobs/index.html) tab.
    

4. **Run `telco-churn-model-train` job**
    - Manually trigger job via UI
        - In the Databricks workspace go to `Workflows` > `Jobs`, where the `telco-churn-model-train` job will be present.
        - Click into telco-churn-model-train and click ‘Run Now’. Doing so will trigger the job on the specified cluster configuration.
    - Alternatively you can trigger the job using the Databricks CLI:
      - `databricks jobs run-now –job-id JOB_ID`
       
    - Model train job steps (`telco-churn-model-train`)
        1. Train improved “new” classifier (RandomForestClassifier - `max_depth=8`)
        1. Register the model. Model version 2 will be registered to stage=None upon successful model training.
        1. **Manual Step**: MLflow Model Registry UI promotion to stage='Staging'
            - Go to Model registry and manually promote model to stage='Staging'

    **ASIDE:** At this point, there should now be two model versions registered in MLflow Model Registry:
        
    - Version 1 (Production): RandomForestClassifier (`max_depth=4`)
    - Version 2 (Staging): RandomForestClassifier (`max_depth=8`)


5. **Run `telco-churn-model-deployment` job (Continuous Deployment)**
    - Manually trigger job via UI
        - In the Databricks workspace go to `Workflows` > `Jobs`, where the `telco-churn-model-deployment` job will be present.
        - Click into telco-churn-model-deployment and click ‘Run Now’. Doing so will trigger the job on the specified cluster configuration. 
    - Alternatively you can trigger the job using the Databricks CLI:
      - `databricks jobs run-now –job-id JOB_ID`
    
    - Model deployment job steps  (`telco-churn-model-deployment`)
        1. Compare new “candidate model” in `stage='Staging'` versus current Production model in `stage='Production'`.
        1. Comparison criteria set through [`model_deployment.yml`](https://github.com/niall-turbitt/e2e-mlops/blob/main/conf/job_configs/model_deployment.yml)
            1. Compute predictions using both models against a specified reference dataset
            1. If Staging model performs better than Production model, promote Staging model to Production and archive existing Production model
            1. If Staging model performs worse than Production model, archive Staging model
            

6. **Run `telco-churn-model-inference-batch` job** 
    - Manually trigger job via UI
        - In the Databricks workspace go to `Workflows` > `Jobs`, where the `telco-churn-model-inference-batch` job will be present.
        - Click into telco-churn-model-inference-batch and click ‘Run Now’. Doing so will trigger the job on the specified cluster configuration.
    - Alternatively you can trigger the job using the Databricks CLI:
      - `databricks jobs run-now –job-id JOB_ID`

    - Batch model inference steps  (`telco-churn-model-inference-batch`)
        1. Load model from stage=Production in Model Registry
            - **NOTE:** model must have been logged to MLflow using the Feature Store API
        1. Use primary keys in specified inference input data to load features from feature store
        1. Apply loaded model to loaded features
        1. Write predictions to specified Delta path

## Limitations
- Multitask jobs running against the same cluster
    - The pipeline initial-model-train-register is a [multitask job](https://docs.databricks.com/data-engineering/jobs/index.html) 
      which stitches together demo setup, feature store creation and model train pipelines. 
    - At present, each of these tasks within the multitask job is executed on a different automated job cluster, 
      rather than all tasks executed on the same cluster. As such, there will be time incurred for each task to acquire 
      cluster resources and install dependencies.
    - As above, we recommend using a pool from which instances can be acquired when jobs are launched to reduce cluster start up time.
    
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

#### Running unit tests

For unit testing, please use `pytest`:
```
pytest tests/unit --cov
```

Please check the directory `tests/unit` for more details on how to use unit tests.
In the `tests/unit/conftest.py` you'll also find useful testing primitives, such as local Spark instance with Delta support, local MLflow and DBUtils fixture.

#### Running integration tests

There are two options for running integration tests:

- On an interactive cluster via `dbx execute`
- On a job cluster via `dbx launch`

For quicker startup of the job clusters we recommend using instance pools ([AWS](https://docs.databricks.com/clusters/instance-pools/index.html), [Azure](https://docs.microsoft.com/en-us/azure/databricks/clusters/instance-pools/), [GCP](https://docs.gcp.databricks.com/clusters/instance-pools/index.html)).

For an integration test on interactive cluster, use the following command:
```
dbx execute --cluster-name=<name of interactive cluster> --job=<name of the job to test>
```

For a test on an automated job cluster, deploy the job files and then launch:
```
dbx deploy --jobs=<name of the job to test> --files-only
dbx launch --job=<name of the job to test> --as-run-submit --trace
```

Please note that for testing we recommend using [jobless deployments](https://dbx.readthedocs.io/en/latest/run_submit.html), so you won't affect existing job definitions.

### Interactive execution and development on Databricks clusters

1. `dbx` expects that cluster for interactive execution supports `%pip` and `%conda` magic [commands](https://docs.databricks.com/libraries/notebooks-python-libraries.html).
2. Please configure your job in `conf/deployment.yml` file.
2. To execute the code interactively, provide either `--cluster-id` or `--cluster-name`.
```bash
dbx execute \
    --cluster-name="<some-cluster-name>" \
    --job=job-name
```

Multiple users also can use the same cluster for development. Libraries will be isolated per each execution context.

### Working with notebooks and Repos

To start working with your notebooks from [Repos](https://docs.databricks.com/repos/index.html), do the following steps:

1. Add your git provider token to your user settings
2. Add your repository to Repos. This could be done via UI, or via CLI command below:
```bash
databricks repos create --url <your repo URL> --provider <your-provider>
```
This command will create your personal repository under `/Repos/<username>/telco_churn`.
3. To set up the CI/CD pipeline with the notebook, create a separate `Staging` repo:
```bash
databricks repos create --url <your repo URL> --provider <your-provider> --path /Repos/Staging/telco_churn
```
