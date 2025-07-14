import subprocess
import sys

# List of script paths to run
scripts = [
    # 'src/data/fetch_hellodata_features.py',
    'src/data/aggregate_hellodata_features.py',
    'src/data/get_location_features.py',
    'src/data/process_survey_data.py',
    # 'src/data/fetch_google_reviews.py',
    'src/data/assemble_final_df.py',
    'src/features/select_features.py',
    'src/models/train_full_aided_model.py',
    # 'src/models/train_market_aided_model.py'
]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)

    print("Output:")
    print(result.stdout)

    if result.stderr:
        print("Errors:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ {script} failed with exit code {result.returncode}")
        break
    else:
        print(f"✅ {script} completed successfully.")