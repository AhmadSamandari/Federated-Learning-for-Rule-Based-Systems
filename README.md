# Federated Learning for Rule-Based Systems

Run federated experiments (Flirt, FedAvg, FedProx) from a single, documented Python entrypoint.

Requirements
- Python 3.8+
- Install project dependencies listed in your environment (TensorFlow for NN models if required).

Files of interest
- main.py — primary runnable script. Use this; the original notebook can be removed.
- configs/ — example YAML configs per model (main.py will pick a default if --config not provided).
- src/ — implementation modules (servers, clients, utils).
- results/ and temp_client_rules/ — outputs created by runs.

Quick examples (from repository root on Windows)

- Run Flirt with 12 clients:
  python main.py --model Flirt --dataset HSP --num-clients 12

- Run FedAvg with 6 clients and explicit config:
  python main.py --model FedAvg --dataset HSP --num-clients 6 --config configs/fedavg_vp_config.yaml

Notes
- num-clients must be one of: 6, 12, 24 (script evenly splits clients between the two data nodes).
- If no --config is provided, main.py attempts to pick a model-appropriate YAML from configs/.
- Ensure your chosen config includes dataset_node1_path and dataset_node2_path keys pointing to CSVs or other supported dataset files.
- The script computes client_sample_size automatically based on train split sizes and requested number of clients.

PowerShell convenience (example)
- From PowerShell:
  python .\main.py --model Flirt --dataset HSP --num-clients 12

If you want, a short main.ps1 wrapper can be added to parse PowerShell parameters and call python main.py.

If anything fails, open an issue and include the full command you used and the config file path.
