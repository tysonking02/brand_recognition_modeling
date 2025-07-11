import pandas as pd
import statsmodels.api as sm

final = pd.read_csv('data/processed/asset_metrics.csv')

with open('data/processed/selected_cols_asset.txt', 'r') as f:
    selected_cols = [line.strip() for line in f]

final = final[selected_cols + ['aided_recognition', 'market']]

for market in final['market'].unique():

    model_name = f"{market} - Aided - Asset"

    cur = final[final['market'] == market]

    x = cur.drop(columns=['aided_recognition', 'market'])
    x = sm.add_constant(x)

    y = cur['aided_recognition']

    model = sm.OLS(y, x)
    results = model.fit()

    with open(f'outputs/model_summaries/{model_name}.txt', 'w') as f:
        f.write(results.summary().as_text())