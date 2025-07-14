import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

final = pd.read_csv('data/processed/manager_metrics.csv')
final = final.loc[:, ~final.columns.str.startswith('Unnamed')]


x = final.drop(columns=['market', 'manager', 'aided_recognition', 'unaided_recognition'])

nan_cols = x.columns[x.isna().any()]
print("Columns with NaNs:", nan_cols.tolist())

y = final['aided_recognition']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

elastic_net = ElasticNetCV(
    alphas=np.logspace(-2, 1, 100),
    l1_ratio=[0.05, .1, .3, .5, .7, .9, .95, .99, 1],  # 1 = Lasso, 0 = Ridge
    cv=5,
    random_state=0,
    max_iter=10000
)
elastic_net.fit(X_train_scaled, y_train)

selected_mask = np.abs(elastic_net.coef_) > 0
selected_cols = X_train.columns[selected_mask].tolist()

if 'manager_brand' not in selected_cols:
    selected_cols += ['manager_brand']

with open('data/processed/selected_cols_manager.txt', 'w') as f:
    for col in selected_cols:
        f.write(f"{col}\n")