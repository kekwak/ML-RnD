from matplotlib import pyplot as plt
import pandas as pd
import mlflow
import os

RUNS = ['bert_frozen', 'baseline', 'bert_len128', 'bert_no_pos_weight', 'bert']

os.makedirs('reports', exist_ok=True)

runs_df = mlflow.search_runs(experiment_names=['jigsaw'], order_by=['start_time DESC'])
runs_df['name'] = runs_df['tags.mlflow.runName'].str.rsplit('-', n=1).str[0]
runs_df = runs_df.drop_duplicates('name').set_index('name').loc[RUNS]

columns = [c for c in runs_df.columns if c.startswith('params.')] + ['metrics.fold0_val_auc']
runs_df[columns].dropna(axis=1, how='all').to_csv('reports/experiments.csv')

scores = runs_df['metrics.fold0_val_auc']
colors = ['#8f8e89' if name == 'baseline' else '#2a78d6' for name in RUNS]

fig, ax = plt.subplots(figsize=(7, 3))
ax.scatter(scores, RUNS, s=70, color=colors, zorder=3)
for name, score in scores.items():
    ax.annotate(f'{score:.4f}', (score, name), xytext=(8, 0), textcoords='offset points', va='center', color='#52514e')
ax.set_xlim(0.93, 1.0)
ax.set_xlabel('ROC-AUC на fold 0')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color='#e5e4df')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('reports/experiments.png', dpi=150)

history = mlflow.MlflowClient().get_metric_history(runs_df.loc['bert', 'run_id'], 'fold0_train_loss')
loss = pd.Series({m.step: m.value for m in history}).sort_index()

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(loss.index, loss.values, color='#2a78d6', alpha=0.25, linewidth=1)
ax.plot(loss.index, loss.rolling(10, center=True, min_periods=1).mean(), color='#2a78d6', linewidth=2)
ax.axvline(loss.index.max() / 2, color='#8f8e89', linestyle='--', linewidth=1)
ax.set_xlabel('шаг')
ax.set_ylabel('train loss')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color='#e5e4df')
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig('reports/bert_train_loss.png', dpi=150)
