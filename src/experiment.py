import time

import mlflow


def setup_experiment(name):
    mlflow.set_experiment(name)


def start_run(run_name=None):
    return mlflow.start_run(run_name=run_name)


def log_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def measure_training_time(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    training_time = time.time() - start
    return result, training_time
