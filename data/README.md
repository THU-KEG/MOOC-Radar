# Data prepare

## Data for baselines reproduction:

1. `--mode` (Option: Coarse/Middle/Fine) for your settings
2. `--data_dir` with Corresponding granularity data from above table.
    
    for example, for `--mode Middle` setting, prepare the following files:
    - `./data/student-problem-middle.json`
    - `./data/problem.json`
3. then generate train/test dataset by setting: `--data_process` in scripts

## Data for improvement reproduction with cognitive and video side information:

Option 1: generate by setting: `--data_process` in scripts

Option 2: download from [there]()