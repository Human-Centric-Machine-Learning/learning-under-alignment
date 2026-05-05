# Learning to Decide with AI-Assistance under Human-Alignment

This repository contains the code and data for the paper 
## Dependencies

### Python
All experiments were performed using Python 3.13.5 To create a virtual environment and install dependencies run:


```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Repository structure

```
├── configs/
│   └── human_ai_interactions.json
│   └── human_alignment.json
├── data/
│   ├── human_ai_interactions_data/
│   └── human_alignment_data/
├── figures/
├── notebooks/
├── outputs/
├── scripts/
│   ├── human_ai_interactions.sh
│   └── human_alignment.sh
└── src/
    ├── bandits.py
    ├── experiment.py
    ├── plot.py
    └── utils.py
```

- `configs/` — JSON files specifying experiment parameters
- `data/` — Raw and processed data for each of the datasets
- `figures/` — Generated figures
- `notebooks/` — Jupyter notebooks for data preprocessing
- `outputs/` — Experiment results (JSON files)
- `scripts/` — Shell scripts to run the experiment for each dataset
- `src/` — Source code:
  - `bandits.py` — Deploys our online learning algorithm, the contextual vanilla baseline and the optimal policy  
  - `experiment.py` — Main experiment loop 
  - `utils.py` — Utility functions to find the optimal decision policy 
  - `plot.py` — Generates plots and legends 

## Running experiments

To run the online learning algorithms and the optimal decision policy for an entire dataset run:

```bash
./scripts/<dataset_name>.sh 
```
where `<dataset_name>` is one of `human_ai_interactions` and `human_alignment`.

### Output

Results are saved to the `output_dir` specified in the config (default: `outputs/<dataset_name>`). If the `DEBUG` flag in `experiment.py`  is set to `True` logs are saved to `log_dir` (default: `outputs/<dataset_name>/logs`) otherwise no logs are printed. The optimal decisions are saved under `optimal_path_actions` (default: `outputs/<dataset_name>/optimal_policies/actions`) specified in the respective config file. To generate the plots used in the paper run: 

```
python3 -m src.plot
```

The plots are saved under `figures/<dataset_name>/<num_iterations>` by default, where `<num_iterations>` is specified in each config file by `num_iterations`.