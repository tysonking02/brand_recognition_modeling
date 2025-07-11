import pandas as pd
import re

# import manager and market maps from survey to costar defaults
with open('data/processed/manager_map.txt', 'r') as f:
    manager_map = eval(f.read())

with open('data/processed/market_map.txt', 'r') as f:
    market_map = eval(f.read())

# region Retrieve Brand Recognition Numbers

# read in raw survey data
raw_survey_data = pd.read_csv('data/raw/raw_survey_data.csv', encoding='latin1').rename(columns={
    'Market': 'market',
    'Which of the following best describes your current living situation?': 'living',
    'What is your combined, annual household income?': 'income',
    'What is theÂ\xa0total monthly rent payment (for all bedrooms)Â\xa0where you live? The total rent forÂ\xa0all bedrooms, not just your portion of the rent.Â\xa0': 'total_rent',
    'What is your age?': 'age',
    'Cortland Unaided': 'cortland_unaided',
    'Camden Unaided': 'camden_unaided',
    'Greystar Unaided': 'greystar_unaided',
    'MAA Unaided': 'maa_unaided'
})

print(f'Processing {len(raw_survey_data)} survey responses...\n')

# create a UID for respondent
raw_survey_data['respondent_id'] = raw_survey_data.index

# change values from Yes and '' to 1 and 0
for col in ['cortland_unaided', 'camden_unaided', 'greystar_unaided', 'maa_unaided']:
    raw_survey_data[col] = raw_survey_data[col].notna().astype(int)

aided_cols = []
unaided_cols = [col for col in raw_survey_data.columns if col.endswith('_unaided')]

# change format of aided columns
for col in raw_survey_data.columns:
    if '<strong>' in col:
        match = re.search(r'<strong>(.*?)</strong>', col)
        if match:
            brand = match.group(1).strip().lower().replace(' ', '_')
            new_col = f"{brand}_aided"
            raw_survey_data[new_col] = raw_survey_data[col].notna().astype(int)
            aided_cols.append(new_col)

survey_df = raw_survey_data[['respondent_id', 'market', 'living', 'income', 'total_rent', 'age'] + aided_cols + unaided_cols]

survey_df.to_csv('data/processed/survey_responses.csv', index=False)

# melt survey results for aided recognition columns
aided_melt = survey_df[['respondent_id', 'market'] + aided_cols].melt(
    id_vars=['respondent_id', 'market'],
    var_name='manager',
    value_name='aided_recognition'
)
aided_melt['manager'] = aided_melt['manager'].str.replace('_aided', '', regex=False)

# melt survey results for unaided recognition columns
unaided_melt = survey_df[['respondent_id', 'market'] + unaided_cols].melt(
    id_vars=['respondent_id', 'market'],
    var_name='manager',
    value_name='unaided_recognition'
)
unaided_melt['manager'] = unaided_melt['manager'].str.replace('_unaided', '', regex=False)

# merge on respondent id
merged = pd.merge(
    aided_melt,
    unaided_melt,
    on=['respondent_id', 'market', 'manager'],
    how='outer'
)

# get average recognition by market/manager combo 
brand_recognition = (
    merged.groupby(['market', 'manager'], as_index=False)
    .agg(
        aided_recognition=('aided_recognition', 'mean'),
        unaided_recognition=('unaided_recognition', 'mean'),
        total_responses=('respondent_id', 'count')
    )
)

# map markets and managers to costar defaults
brand_recognition['market'] = brand_recognition['market'].map(market_map)
brand_recognition['manager'] = brand_recognition['manager'].map(manager_map)

brand_recognition.dropna(subset=['market', 'manager'], inplace=True)

brand_recognition.to_csv('data/processed/brand_recognition.csv', index=False)

# endregion