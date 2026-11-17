from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from src.train.utils import setup_run, get_folds, evaluate_fold, save_results
from loguru import logger
import mlflow
import pandas as pd
import numpy as np
import importlib
import joblib
import json

def main():
    config, RUN_PATH = setup_run()

    logger.info('Loding variables')

    SEED = config['seed']
    TEXT_COLUMN = config['text_column']
    LABEL_COLUMNS = config['label_columns']

    TRAIN_CSV_PATH = config['train_csv_path']
    TEST_CSV_PATH = config['test_csv_path']

    vec_module_path, vec_class_name = config['vectorizer']['class'].rsplit(".", 1)
    vec_module = importlib.import_module(vec_module_path)
    vec_kwargs = config['vectorizer'].get('params') or {}

    clf_module_path, clf_class_name = config['clf_model']['class'].rsplit(".", 1)
    clf_module = importlib.import_module(clf_module_path)
    clf_kwargs = config['clf_model'].get('params') or {}
    clf_class = getattr(clf_module, clf_class_name)
    if 'random_state' in clf_class().get_params():
        clf_kwargs.setdefault('random_state', SEED)

    logger.info('Reading data')
    df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    folds = get_folds(df, config)
    oof_pred = np.zeros((len(df), len(LABEL_COLUMNS)))
    test_pred = np.zeros((len(test_df), len(LABEL_COLUMNS)))
    oof_idx = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in folds:
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        logger.info(f'Fold {fold}: training pipe')
        pipe = Pipeline([
            ('tf-idf', getattr(vec_module, vec_class_name)(**vec_kwargs)),
            ('clf', OneVsRestClassifier(clf_class(**clf_kwargs)))
        ]).fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMNS])

        logger.info(f'Fold {fold}: making val predictions')
        oof_pred[val_idx] = pipe.predict_proba(val_df[TEXT_COLUMN])
        oof_idx.append(val_idx)

        fold_metrics.append(evaluate_fold(val_df, oof_pred[val_idx], LABEL_COLUMNS))
        mlflow.log_metric(f'fold{fold}_val_auc', fold_metrics[-1]['avg_score'])
        logger.info(f'Fold {fold} metrics = {json.dumps(fold_metrics[-1], indent=2)}')

        logger.info(f'Fold {fold} making test predictions')
        test_pred += pipe.predict_proba(test_df[TEXT_COLUMN]) / len(folds)

        joblib.dump(pipe, f'{RUN_PATH}/model_fold{fold}.pkl')

    save_results(RUN_PATH, config, df, test_df, oof_pred, test_pred, oof_idx, fold_metrics)

if __name__ == '__main__':
    main()
