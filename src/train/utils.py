from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.evaluation.evaluate import evaluate
from loguru import logger
import pandas as pd
import numpy as np
import mlflow
import random
import argparse
import time
import json
import yaml
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def setup_run():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    np.random.seed(config['seed'])
    random.seed(config['seed'])

    run_name = f'{config['run_name']}-{time.time_ns()}'
    run_path = f'./runs/{run_name}'

    os.makedirs(run_path, exist_ok=True)
    logger.add(f'{run_path}/train.log')
    shutil.copy(args.config, f'{run_path}/{args.config.split('/')[-1]}')

    mlflow.set_experiment('jigsaw')
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(pd.json_normalize(config).iloc[0].to_dict())

    logger.info(f'Start training with config: {json.dumps(config, indent=2)}')
    return config, run_path

def get_folds(df, config):
    mskf = MultilabelStratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
    folds = config.get('folds') or range(config['n_splits'])
    return [(fold, idx) for fold, idx in enumerate(mskf.split(df, df[config['label_columns']])) if fold in folds]

def evaluate_fold(val_df, val_pred, label_columns):
    pred_val_df = pd.DataFrame(val_pred, columns=label_columns)
    pred_val_df['id'] = val_df['id'].values
    return evaluate(val_df, pred_val_df)

def save_results(run_path, config, df, test_df, oof_pred, test_pred, oof_idx, fold_metrics):
    label_columns = config['label_columns']

    fold_metrics_df = pd.DataFrame(fold_metrics)
    metrics = {
        'mean': fold_metrics_df.mean().to_dict(),
        'std': fold_metrics_df.std().to_dict(),
        'folds': fold_metrics,
    }
    logger.info(f'CV done, mean = {json.dumps(metrics["mean"], indent=2)}, std = {json.dumps(metrics["std"], indent=2)}')

    oof_df = pd.DataFrame(oof_pred, columns=label_columns)
    oof_df.insert(0, 'id', df['id'].values)
    oof_df = oof_df.iloc[np.sort(np.concatenate(oof_idx))]
    submission_df = pd.DataFrame(test_pred, columns=label_columns)
    submission_df.insert(0, 'id', test_df['id'].values)

    logger.info('Saving artifacts')
    with open(f'{run_path}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    oof_df.to_csv(f'{run_path}/oof_predictions.csv', index=False)
    submission_df.to_csv(f'{run_path}/submission.csv', index=False)

    mlflow.log_metrics({f'cv_{k}': v for k, v in metrics['mean'].items()})
    mlflow.log_artifact(f'{run_path}/metrics.json')
    mlflow.log_artifact(f'{run_path}/train.log')
    mlflow.end_run()

    logger.info('Training done')
