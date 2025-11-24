# Federated-Learning-for-Rule-Based-Systems

Lightweight repository for running federated experiments (Flirt, FedAvg, FedProx) with rule-based server logic and NN baselines.

## Quick overview
- Notebooks: [main1.ipynb](main1.ipynb) (exploratory).
- Script entrypoint (preferred for reproducible runs): [main.py](main.py) — call with a config file.
- Configs live under `configs/` (examples below).
- Results saved to `results/` and temporary client rule files in `temp_client_rules/`.

## Run with Python (recommended)
Choose a config and run the script:
```sh
python main.py --config configs/flirt_config_vp.yaml
python main.py --config configs/fedavg_vp_config.yaml
python main.py --config configs/fedprox_config.yaml
```
Default notebook entry also accepts `--config`. See [main1.ipynb](main1.ipynb) for notebook usage.

## Models & how to choose
- Flirt (rule-based server + synthetic labeling): configuration examples:
  - [configs/flirt_config_vp.yaml](configs/flirt_config_vp.yaml)
  - checkpoint: [configs/.ipynb_checkpoints/flirt_config_vp-checkpoint.yaml](configs/.ipynb_checkpoints/flirt_config_vp-checkpoint.yaml)
  Server class: [`src.flirt.server.FlirtServer`](src/flirt/server.py)  
  Client class: [`src.flirt.client.FlirtClient`](src/flirt/client.py)

- FedAvg (neural-net aggregator):
  - [configs/fedavg_vp_config.yaml](configs/fedavg_vp_config.yaml)
  Server class: [`src.fedavg.server.FedAvgServer`](src/fedavg/server.py)  
  Client class: [`src.fedavg.client.FedAvgClient`](src/fedavg/client.py)

- FedProx (FedAvg variant with proximal term):
  - [configs/fedprox_config.yaml](configs/fedprox_config.yaml)
  Server class: [`src.fedprox.server.FedProxServer`](src/fedprox/server.py)  
  Client class: [`src.fedprox.client.FedProxClient`](src/fedprox/client.py)

## Useful modules
- Data utilities: [`src.utils.data_utils`](src/utils/data_utils.py)
- Model utilities: [`src.utils.model_utils`](src/utils/model_utils.py)
- Rule utilities: [`src.utils.rule_utils`](src/utils/rule_utils.py)

## Config tips
- Each config sets dataset paths, number of rounds, client/sample sizes, learning rates, output filename (see `output_filename`).
- To switch model, change `model_name` in a YAML config or pass a different config file.

## Outputs
- Numeric results saved under `results/` (e.g. `results/flirt_results_vp.npy`).
- Temporary client rule files: `temp_client_rules/`.

## Reproducibility
- Seeds are respected via `random_state` in configs (see [main.py](main.py) and [main1.ipynb](main1.ipynb)).
- For Flirt, server training and synthetic data generation controlled in [`src.flirt.server.FlirtServer.run_federated_training`](src/flirt/server.py).

## Troubleshooting
- If running as script, ensure project root is on PYTHONPATH (the notebook inserts `../` automatically).
- If a column `Unnamed: 0` appears, utils drop it automatically in data loaders (`src.utils.data_utils`).

Contributions and issues: open PRs/issues referencing files above.
