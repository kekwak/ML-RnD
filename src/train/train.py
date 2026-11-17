from sklearn.multiclass import OneVsRestClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.pipeline import Pipeline
from src.evaluation.evaluate import evaluate
from loguru import logger
import pandas as pd
import numpy as np
import random
import importlib
import argparse
import joblib
import time
import json
import yaml
import shutil
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    SEED = config['seed']
    np.random.seed(SEED)
    random.seed(SEED)

    RUN_NAME = f'{config['run_name']}-{time.time_ns()}'
    RUN_PATH = f'./runs/{RUN_NAME}'

    os.makedirs(RUN_PATH, exist_ok=True)
    logger.add(f"{RUN_PATH}/train.log")
    shutil.copy(args.config, f'{RUN_PATH}/{args.config.split('/')[-1]}')

    logger.info('Loding variables')

    TEXT_COLUMN = config['text_column']
    LABEL_COLUMNS = config['label_columns']

    TRAIN_CSV_PATH = config['train_csv_path']
    TEST_CSV_PATH = config['test_csv_path']
    N_SPLITS = config['n_splits']

    vec_module_path, vec_class_name = config['vectorizer']['class'].rsplit(".", 1)
    vec_module = importlib.import_module(vec_module_path)
    vec_kwargs = config['vectorizer'].get('params') or {}

    clf_module_path, clf_class_name = config['clf_model']['class'].rsplit(".", 1)
    clf_module = importlib.import_module(clf_module_path)
    clf_kwargs = config['clf_model'].get('params') or {}
    clf_class = getattr(clf_module, clf_class_name)
    if 'random_state' in clf_class().get_params():
        clf_kwargs.setdefault('random_state', SEED)

    logger.info(f'Start training with config: {json.dumps(config, indent=2)}')

    logger.info('Reading data')
    df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_pred = np.zeros((len(df), len(LABEL_COLUMNS)))
    test_pred = np.zeros((len(test_df), len(LABEL_COLUMNS)))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(mskf.split(df, df[LABEL_COLUMNS])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        logger.info(f'Fold {fold}: training pipe')
        pipe = Pipeline([
            ('tf-idf', getattr(vec_module, vec_class_name)(**vec_kwargs)),
            ('clf', OneVsRestClassifier(clf_class(**clf_kwargs)))
        ]).fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMNS])

        logger.info(f'Fold {fold}: making val predictions')
        oof_pred[val_idx] = pipe.predict_proba(val_df[TEXT_COLUMN])

        pred_val_df = pd.DataFrame(oof_pred[val_idx], columns=LABEL_COLUMNS)
        pred_val_df['id'] = val_df['id'].values
        fold_metrics.append(evaluate(val_df, pred_val_df))
        logger.info(f'Fold {fold} metrics = {json.dumps(fold_metrics[-1], indent=2)}')

        logger.info(f'Fold {fold} making test predictions')
        test_pred += pipe.predict_proba(test_df[TEXT_COLUMN]) / N_SPLITS

        joblib.dump(pipe, f'{RUN_PATH}/model_fold{fold}.pkl')

    fold_metrics_df = pd.DataFrame(fold_metrics)
    metrics = {
        'mean': fold_metrics_df.mean().to_dict(),
        'std': fold_metrics_df.std().to_dict(),
        'folds': fold_metrics,
    }
    logger.info(f'CV done, mean = {json.dumps(metrics["mean"], indent=2)}, std = {json.dumps(metrics["std"], indent=2)}')

    oof_df = pd.DataFrame(oof_pred, columns=LABEL_COLUMNS)
    oof_df.insert(0, 'id', df['id'].values)
    submission_df = pd.DataFrame(test_pred, columns=LABEL_COLUMNS)
    submission_df.insert(0, 'id', test_df['id'].values)

    logger.info('Saving artifacts')
    with open(f'{RUN_PATH}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    oof_df.to_csv(f'{RUN_PATH}/oof_predictions.csv', index=False)
    submission_df.to_csv(f'{RUN_PATH}/submission.csv', index=False)

    logger.info('Training done')

if __name__ == '__main__':
    main()
