# Unimodal Bandits for Pricing and Pseudo-collusion due to Localised Exploration


The code supports the research presented in Chapter 4 of the PhD thesis:  
“Algorithmic Pricing in Multi-agent, Competitive Markets”  
by Lukasz Sliwinski, 2025, University of Edinburgh.

---

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install in editable mode:

```bash
pip install -e .
```

## Runnning the experiments

The source code with definitions of the bandits and the competitions is located within the `src` directory.

The experiments are located in three main directories:
- EXPERIMENT_ANALYSIS_000: Contains the analysis of the unimodal bandit algorithms and the hyperparameter optimisation study.
- EXPERIMENT_COMPETITION: Contains the competition simulations between unimodal bandits with parameters obtained in the hyperparameter optimisation.
- EXPERIMENT_COMPETITION_STATIONARY: Contains the competition simulation between unimodal bandits optimised for more stationary competitions.

Because the code is installed as a package, all scripts should be run from their parent folder i.e.
```python
cd experiments/EXPERIMENT_ANALYSIS_000/_scripts
python tuning_run.py
```

### EXPERIMENT_ANALYSIS_000

Most of the scripts are located in the `_scripts` folder. 

- `sample_plot_competitions.py` computes the sample evaluation of the algorithm against the testcases (stationary/nonstationary). Change the algorithm in the __main__ block.
- `tuning_sample_definition.py` creates the hyperparameter grid definition for tuning - it needs to be set once per experiment and not altered!
- `tuning_run.py` runs the tuning for a given algorithm and range of noises  and outputs the evaluation (regret) to appriopriate folder. Change the algorithm in the __main__ block.
- `analysis_combine_noise.py` averages evaluation across different noise levels and produes a new evaluation dataframe
- `analysis_pgp_gp.py` runs the Partial Dependence Plot analysis using Gaussian processes for given algorithm and metric i.e. regret averaged over all testcases and noises
- `evaluate_algorithms_fixed_params.py` evaluates the algorithm with fixed params against all specified testcases and noises. This is used to get unbiased final score of algorithms with given parameters. Change the algorithm type in the __main__ block.


After running the evaluation of the algorithms, there is a script `algorithm_evaluations/analyse_evaluation.py` which can be used to get a table that compares all the algorithms.


### EXPERIMENT_COMPETITION

To run competitions between two agents with optimised parameters, simply run:
```python 
python competitions_run.py
```
which will run all the competition cases defined in `cases_definition.py`. The agent parameters used for those competitions are in the `agent_defition.py` file.

To analyse the results:
- run `analyse_competitions_1.py` to analyse competitions with one agent only (stationary environment)
- run `analyse_competitions_2.py` to analyse competitions between two agents of the same type.
- run `analyse_competitions_3.py` to analyse competitions between three agents of the same type.

Additionally, run `sample_competition_plots.py` to plot sample competitions (price/reward evolution) for all the competition cases.

### EXPERIMENT_COMPETITION_STATIONARY

This folder is identical in structure to the above but there are no competitions between 3 agents or code to plot sample competitions.
The main difference is the contents of `agent_definition.py` which redefines the agent parameters to be more suitable for a stationary environment.


