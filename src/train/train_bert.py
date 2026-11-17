from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from src.train.utils import setup_run, get_folds, evaluate_fold, save_results
from loguru import logger
import pandas as pd
import numpy as np
import mlflow
import torch
import json

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for enc, _ in loader:
            preds.append(torch.sigmoid(model(**enc.to('cuda')).logits.float()).cpu())
    return torch.cat(preds).numpy()

def main():
    config, RUN_PATH = setup_run()
    torch.manual_seed(config['seed'])

    TEXT_COLUMN = config['text_column']
    LABEL_COLUMNS = config['label_columns']

    TRAIN_CSV_PATH = config['train_csv_path']
    TEST_CSV_PATH = config['test_csv_path']

    MODEL_NAME = config['model']['name']
    MAX_LEN = config['model']['max_len']
    FREEZE_ENCODER = config['model']['freeze_encoder']
    BATCH_SIZE = config['train']['batch_size']
    EPOCHS = config['train']['epochs']
    LR = config['train']['lr']
    POS_WEIGHT = config['train']['pos_weight']

    logger.info('Reading data')
    df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt')
        return enc, torch.tensor(np.array(labels), dtype=torch.float)

    def make_loader(texts, labels, shuffle=False):
        return DataLoader(list(zip(texts, labels)), batch_size=BATCH_SIZE, shuffle=shuffle, collate_fn=collate)

    test_loader = make_loader(test_df[TEXT_COLUMN].values, np.zeros((len(test_df), len(LABEL_COLUMNS))))

    folds = get_folds(df, config)
    oof_pred = np.zeros((len(df), len(LABEL_COLUMNS)))
    test_pred = np.zeros((len(test_df), len(LABEL_COLUMNS)))
    oof_idx = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in folds:
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_loader = make_loader(train_df[TEXT_COLUMN].values, train_df[LABEL_COLUMNS].values, shuffle=True)
        val_loader = make_loader(val_df[TEXT_COLUMN].values, val_df[LABEL_COLUMNS].values)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_COLUMNS)).cuda()
        if FREEZE_ENCODER:
            for p in model.base_model.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        total_steps = EPOCHS * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

        pos = train_df[LABEL_COLUMNS].values.mean(axis=0)
        pos_weight = np.sqrt((1 - pos) / pos) if POS_WEIGHT else np.ones(len(LABEL_COLUMNS))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float).cuda())

        for epoch in range(EPOCHS):
            logger.info(f'Fold {fold} epoch {epoch}: training')
            model.train()
            for step, (enc, labels) in enumerate(train_loader):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = loss_fn(model(**enc.to('cuda')).logits.float(), labels.cuda())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % 100 == 0:
                    mlflow.log_metric(f'fold{fold}_train_loss', loss.item(), step=epoch * len(train_loader) + step)

            val_pred = predict(model, val_loader)
            metrics = evaluate_fold(val_df, val_pred, LABEL_COLUMNS)
            mlflow.log_metric(f'fold{fold}_val_auc_by_epoch', metrics['avg_score'], step=epoch)
            logger.info(f'Fold {fold} epoch {epoch} metrics = {json.dumps(metrics, indent=2)}')

        oof_pred[val_idx] = val_pred
        oof_idx.append(val_idx)
        fold_metrics.append(metrics)
        mlflow.log_metric(f'fold{fold}_val_auc', metrics['avg_score'])

        logger.info(f'Fold {fold} making test predictions')
        test_pred += predict(model, test_loader) / len(folds)

    save_results(RUN_PATH, config, df, test_df, oof_pred, test_pred, oof_idx, fold_metrics)

if __name__ == '__main__':
    main()
