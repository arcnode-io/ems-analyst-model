
# EMS Analyst Model 📈🧠
![](https://img.shields.io/badge/timescale-gray?logo=timescale)
![](https://img.shields.io/badge/mlflow-gray?logo=mlflow)
![](https://img.shields.io/badge/validator-pandera-78ac1b)

> Solar supply forecasting model training and MLflow deployment - comparing Prophet vs LSTM vs XGBoost on hourly data since 2016

## Prerequisites
- Ercot API  Account


## Activity Diagram for Training

The following logic ensures:

- Best performing model is always selected

- System never silently degrades below baseline

- Human intervention is triggered when needed

```plantuml
if (challenger > champion) then (yes)
 :deploy challenger;
 stop
else (no)
 if (baseline > champion) then (yes)
   :send grafana alert;
   :deploy baseline;
   stop
 else (no)
   :keep champion;
   stop
```

### Deployment Diagram

```plantuml
cloud ercot_api
rectangle train
database timeseries
rectangle model_store
ercot_api - train: solar production data
train - timeseries: hourly data since 2016
train -d- model_store: deploy MLflow model
```

## Sequence Diagram

The CI versioning job should version the new model in MLflow for the server workspace member to consume

```plantuml
collections ercot_api
participant process
database timeseries
participant train
participant grafana_prometheus
actor data_scientist
collections ci_versioning_job
database model_store
== Daily Automated Training ==
process -> ercot_api: request solar data
process -> timeseries: load hourly data
timeseries <- train: get training data
train -> train: compare Prophet vs LSTM vs XGBoost
train -> train: promote best model
train -> model_store: deploy MLflow model

== Manual Data Scientist Retraining ==
train -> grafana_prometheus: send alert
grafana_prometheus -> data_scientist: receives w/ Grafana Notifier
data_scientist -> train: train new challenger
data_scientist -> ci_versioning_job: breaking change conventional commit
ci_versioning_job -> model_store: version new MLflow model
```

> *🐢 will setup webhook when latency is high

## Project Structure

```
├── pyproject.toml           # Dependencies and build config
├── model_selection.ipynb    # Model comparison analysis and selection
├── seed.py                  # Script for seeding historical data
├── src/
│   ├── main.py              # Training pipeline entry point
│   ├── models.py            # Model definitions and training
│   ├── process.py           # Data processing pipeline
│   ├── process_test.py      # Data processing tests
│   ├── utils.py             # Utility functions
│   └── utils_test.py        # Utility tests
├── tests/
│   └── test_integration.py  # End-to-end model tests
└── readme.md                # This file
```
