# !/bin/bash

# activate the virtual environment
source .venv/bin/activate

# This is a bash script that runs all jobs from 0 to 23
# Each job corresponds to a different algorithm and dataset group pair
for job_id in {0..23}
do
    echo "Running job $job_id"
    python src/experiment.py --config_file configs/human_alignment.json --job_id $job_id
done

