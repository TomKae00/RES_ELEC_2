# RES_ELEC_2

This repository contains the code developed for the second assignment in **Renewables in Electricity Markets**.

The assignment is structured into two main parts:

* **Task 1:** Day-ahead offering strategies for a wind producer under balancing market uncertainty.
* **Task 2:** Reserve bidding under load uncertainty using chance-constrained and CVaR-based formulations.

Each task script can be run independently and writes its results to a dedicated output folder.

## Project structure

The repository is organized as follows:

```
RES_ELEC_2/
├── data/
├── outputs/
├── helpers.py
├── helpers2.py
├── plotting.py
├── analyze_scenarios.py
├── task_1_1.py
├── task_1_2.py
├── task_1_3.py
├── task_1_4.py
├── task_2_1.py
├── task_2_2.py
└── task_2_3.py
```

## Main task scripts

### Task 1: Wind offering strategy

* `task_1_1.py`
  Solves the day-ahead offering problem under a one-price balancing scheme.

* `task_1_2.py`
  Solves the day-ahead offering problem under a two-price balancing scheme and compares it to the one-price case.

* `task_1_3.py`
  Performs in-sample and out-of-sample cross-validation for the one-price and two-price strategies.

* `task_1_4.py`
  Extends the offering model with a CVaR-based risk term.

### Task 2: Reserve bidding

* `task_2_1.py`
  Generates load profiles and solves the reserve bidding problem using ALSO-X and CVaR formulations.

* `task_2_2.py`
  Evaluates the Task 2.1 reserve bids out-of-sample.

* `task_2_3.py`
  Performs a sensitivity analysis over different reliability thresholds.

## Shared files

* `helpers.py`
  Contains helper functions for Task 1, including scenario loading, preprocessing, scenario construction, and profit evaluation.

* `helpers2.py`
  Contains helper functions for Task 2, including load-profile generation and reserve-availability calculations.

* `plotting.py`
  Contains all plotting functions used across Task 1 and Task 2.

* `analyze_scenarios.py`
  Creates diagnostic plots and tables for the Task 1 scenario data.

## Data

The main input data are stored in:

```
data/
├── scen_zone2.csv
├── DayAheadPrices.csv
└── processed/
```

The raw input files are:

* `data/scen_zone2.csv`
  Wind production scenarios.

* `data/DayAheadPrices.csv`
  Day-ahead price data.

The `data/processed/` folder contains processed diagnostic files created during preprocessing.

## Outputs

All generated results are stored in:

```
outputs/
├── task_1/
│   ├── task_1_1/
│   ├── task_1_2/
│   ├── task_1_3/
│   └── task_1_4/
├── task_2/
│   ├── task_2_1/
│   ├── task_2_2/
│   └── task_2_3/
└── scenario_analysis/
```

Each task writes its own tables and figures into the corresponding folder.

## General code structure

The scripts follow a common structure:

1. **Input preparation**
   Data and scenarios are loaded or generated.

2. **Model formulation**
   The optimization problem is built in Gurobi.

3. **Model solving**
   The optimization model is solved.

4. **Result evaluation**
   Profits, reserve bids, reliability metrics, or risk metrics are calculated.

5. **Output generation**
   Tables and figures are saved to the corresponding output folder.

Each task script is intended to be executable on its own. There is no central `main.py` file for the full repository.

## Requirements

The project dependencies are listed in:

```
requirements.txt
```

Install them with:

```
pip install -r requirements.txt
```

The optimization models require a working Gurobi installation and license.

## Running the scripts

Run scripts from the repository root.

Recommended order:

```
python analyze_scenarios.py

python task_1_1.py
python task_1_2.py
python task_1_3.py
python task_1_4.py

python task_2_1.py
python task_2_2.py
python task_2_3.py
```

Some scripts use outputs from previous tasks:

* `task_1_2.py` uses the Task 1.1 offer file for comparison plots.
* `task_2_2.py` uses the Task 2.1 reserve bid results for out-of-sample verification.
