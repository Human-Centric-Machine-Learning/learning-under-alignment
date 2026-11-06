# !/bin/bash

# activate the virtual environment
source .venv/bin/activate

# This is a bash script that runs all jobs from 0 to 19
# Each job corresponds to a different algorithm and dataset task pair
for job_id in {0..19}
do
    echo "Running job $job_id"
    python src/experiment.py --config_file configs/human_ai_interactions.json --job_id $job_id
done

