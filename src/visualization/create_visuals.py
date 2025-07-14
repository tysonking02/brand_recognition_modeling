import pickle
import matplotlib.pyplot as plt
import pandas as pd

color_map = {
    'manager_brand': '#3297ac',
    'unit_count': '#3297ac',
    'style_Hi-Rise': '#de3d59',
    'property_quality': '#de3d59',
    'building_age': '#de3d59',
    'miles_from_city_center': '#918bbd',
    'pop_density': '#918bbd',
    'years_since_reno': '#de3d59',
    'years_since_first_acquisition': '#3297ac',
    'years_since_acquisition': '#3297ac'
}

# Load model
with open('models/All Markets - Aided - Manager.pkl', 'rb') as f:
    model = pickle.load(f)

# Get coefficients and p-values
coefs = model.params.drop('const')
pvals = model.pvalues.drop('const')

# Keep only significant features (p < 0.1)
# significant = pvals[pvals < 0.1].index
significant = pvals.index
coefs = coefs[significant]

coefs['pop_density'] = coefs['pop_density'] / 43560
coefs['property_quality'] = coefs['property_quality'] / 10
coefs['unit_count'] = coefs['unit_count'] * 1000

# Sort by absolute value for visualization
coefs = coefs.reindex(coefs.sort_values().index)

colors = [color_map.get(feat, '#cccccc') for feat in coefs.index] 

plt.figure(figsize=(10, 8))
plt.barh(coefs.index, coefs.values, color=colors)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([])
plt.yticks(fontsize=12)

plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Standardized Coefficient')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('outputs/figures/feature_importance.png')