import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV

def select_features(interaction):

    if interaction == True:
        print('Selecting cols (with interaction effects)')
    else:
        print('Selecting cols (w/o interaction effects)')

    final = pd.read_csv('data/processed/manager_metrics.csv')
    final = final.loc[:, ~final.columns.str.startswith('Unnamed')]

    X = final.drop(columns=['market', 'manager', 'aided_recognition', 'unaided_recognition'])
    y = final['aided_recognition']

    if interaction:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        X_used = X_poly
    else:
        X_used = X.values
        feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(
        X_used, y, test_size=0.2, random_state=100
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    elastic_net = ElasticNetCV(
        alphas=np.logspace(-2, 1, 100),
        l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
        cv=5,
        random_state=0,
        max_iter=10000
    )
    elastic_net.fit(X_train_scaled, y_train)

    selected_mask = np.abs(elastic_net.coef_) > 0
    selected_features = np.array(feature_names)[selected_mask].tolist()

    if 'manager_brand' not in selected_features:
        selected_features.append('manager_brand')

    output_file = (
        'data/processed/selected_cols_manager_interaction_poly.txt'
        if interaction else
        'data/processed/selected_cols_manager.txt'
    )
    with open(output_file, 'w') as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

select_features(interaction=True)
select_features(interaction=False)