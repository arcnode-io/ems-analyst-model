
# EMS Analyst Model 📈🧠

![](https://img.shields.io/gitlab/pipeline-status/arcnode-io/ems-analyst-model?branch=main&logo=gitlab)
![](https://gitlab.com/arcnode-io/ems-analyst-model/badges/main/coverage.svg)
![](https://img.shields.io/badge/ty_checked-gray?logo=astral)
![](https://img.shields.io/badge/timescale-gray?logo=timescale)
![](https://img.shields.io/badge/mlflow-gray?logo=mlflow)
![](https://img.shields.io/badge/validator-pandera-78ac1b)

> ERCOT day-ahead settlement point price forecasting for HB_NORTH — comparing Prophet, XGBoost, and LightGBM on hourly data backfilled from 2022.

## Prerequisites
- Gridstatus.io API account (`GRIDSTATUS_API_KEY`)
- PostgreSQL time-series database (`TIMESERIES_URL`) and MLflow tracking server

The daily pipeline pulls `ercot_spp_day_ahead_hourly` prices and North-zone load forecasts, trains the models, registers the lowest-holdout-MAE model in MLflow, and writes the next 24 hours of `dam_lmp_price` forecasts in `usd_per_mwh` to PostgreSQL. The analyst server reads the `forecasts` table for the agent and HMI; MLflow stays off the runtime serving path.


## Activity Diagram for Training

Each scheduled run selects the model with the lowest holdout MAE and records the model and baseline metrics in MLflow and Prometheus. The baseline is a comparison metric; it is not deployed as a fallback.

```plantuml
:train Prophet, XGBoost, and LightGBM;
:select lowest holdout MAE;
:register champion in MLflow;
:score next 24 hours;
:write forecasts to PostgreSQL;
stop
```

### Deployment Diagram

```plantuml
cloud gridstatus_api
rectangle train
database timeseries
rectangle model_store
train -l-> gridstatus_api: HTTP\n prices + load forecasts
train -u-> timeseries: SQL\n gets historical
train -d-> model_store: MLflow API\ndeploy model
```

## Sequence Diagram

The daily pipeline registers the champion in MLflow and publishes forecast rows for the analyst server to consume.

```plantuml
collections gridstatus_api
participant process
database timeseries
participant train
participant grafana_prometheus
actor data_scientist
database model_store
== Daily Automated Training ==
process -> gridstatus_api: request DAM prices + load forecasts
process -> timeseries: load hourly data
train -> timeseries: get training data
train -> train: compare Prophet vs XGBoost vs LightGBM
train -> train: promote best model
train -> model_store: register MLflow model version
train -> timeseries: write 24-hour price forecasts

== Manual Data Scientist Retraining ==
train -> grafana_prometheus: expose model + baseline metrics
grafana_prometheus -> data_scientist: receives w/ Grafana Notifier
data_scientist -> train: train new challenger
train -> model_store: register MLflow model version
train -> timeseries: write price forecasts
```

## Project Structure

```
├── pyproject.toml           # Dependencies and build config
├── model_evaluation.ipynb   # Model comparison analysis and evaluation
├── seed.py                  # Script for seeding historical data
├── src/
│   ├── main.py              # Training pipeline entry point
│   ├── app.py               # Scheduling and pipeline orchestration
│   ├── models.py            # Schemas, feature columns, and hyperparameters
│   ├── train.py             # Model training and champion selection
│   ├── score.py             # Forecast generation and PostgreSQL publishing
│   ├── process.py           # Data processing pipeline
│   ├── test_process.py      # Data processing tests
│   ├── utils.py             # Utility functions
│   └── test_utils.py        # Utility tests
├── tests/
│   └── test_integration.py  # End-to-end model tests
└── readme.md                # This file
```
