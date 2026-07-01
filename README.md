# Numerical-Stability-of--Recurrence-Reinforcement-Learning-
study on the stability of reinforcement learning with LSTM in a partial observable environment. 

file structure 

project/
│
├── environment/
│   └── environment.py
│
├── agents/
│   ├── feedforward/
│   │   ├── agent.py
│   │   └── config.py
│   └── lstm_ppo/
│       ├── agent.py
│       └── config.py
│
├── analysis/
│   ├── jacobian.py          # spectral radius computation
│   ├── metrics.py           # grad norms, hidden state drift, reward curves
│   └── statistics.py        # variance, tail analysis, seed aggregation
│
├── experiments/
│   ├── run_experiment.py    # single seed entry point
│   └── run_all_seeds.py     # launches N seeds, aggregates results
│
├── output/
│   ├── runs/
│   │   └── {agent}_{seed}/
│   │       ├── spectral_radius.npy
│   │       ├── grad_norms.npy
│   │       ├── hidden_drift.npy     # LSTM only
│   │       └── rewards.npy
│   ├── figures/
│   │   ├── spectral_radius_variance.png
│   │   ├── distribution_tails.png
│   │   └── reward_vs_instability.png
│   └── summary/
│       └── results_table.csv
│
├── notebooks/
│   └── analysis.ipynb       # exploratory, plotting
│
├── configs/
│   └── default.yaml         # seeds, intervals, hyperparams
│
└── README.md