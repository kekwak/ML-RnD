from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse
import json

TEST_LABELS_CSV_PATH = "data/test_labels.csv"
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission')
    return parser.parse_args()

def evaluate(test_df, submission_df):
    raw_scores = roc_auc_score(test_df[LABELS], submission_df[LABELS], average=None)
    raw_scores_dict = {k: v.item() for k, v in zip(LABELS, raw_scores)}
    raw_scores_dict.update({'avg_score': (sum(raw_scores) / len(raw_scores)).item()})
    return raw_scores_dict

def main():
    args = parse_args()
    test_df = pd.read_csv(TEST_LABELS_CSV_PATH)
    mask = test_df['toxic'] != -1
    test_df = test_df[mask]

    submission_df = pd.read_csv(args.submission)
    submission_df = submission_df[submission_df.id.isin(test_df.id)]
    submission_df = submission_df.set_index('id').loc[test_df.id].reset_index()
    
    metrics = evaluate(test_df, submission_df)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
