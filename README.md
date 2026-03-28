# UAV-FL-MARL

This repository contains the code for joint UAV altitude control and client selection for reliability-aware and fair federated learning in cellular-connected UAV networks.

## Overview

In cellular-connected UAV federated learning, unreliable air-to-ground wireless links can bias training because UAVs with better channel conditions are more likely to successfully upload their local model updates. This project studies that problem and implements multiple client-selection strategies, including a multi-agent reinforcement learning (MARL) approach that jointly controls UAV altitude and client selection.

This repository also provides the modified environment components used in our UAV-FL project. The implementation is designed to work with the [light_mappo](https://github.com/tinyzqh/light_mappo) framework and adapts the environment logic for the UAV federated learning setting.

```bash
python main.py --env highrise --total_UE 20 --active_UE 10 --method marl --marl_mode full --round 100 --dataset mnist --model cnn60k