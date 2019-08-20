### Behavior Cloning

| Task  | Reward (Mean) | Reward (Std.) | Expert Policy (Mean) | Expert Policy (Std) | 
| ------ | ------ | ------ | ------ | ------ | 
| Ant-v2    | 4823.67 | 85.80 | 4717.22 | 124.05 |
| Humanoid-v2 | 2925.325 | 2097.77 | 10681.43 | 34.58 |

| Network Size | Number of Rollouts | Epochs | Batch Size | Timesteps per Rollout
| ------ | ------ | ------ | ------ | ------- |
| (64, 64) | 20 | 100 | 32 | 1000

#### Hyperparameter Analysis: Number of Rollouts (Humanoid-v2)
| Number of Rollouts | Reward (Mean) | Reward (Std) |
| ------ | ------ | ------ | 
| 1 | 217.60 | 0.0 |
| 2 | 236.011 | 3.71 |
| 3 | 263.52 | 43.40 |
| 4 | 352.80 | 70.27 |
| 5 | 396.12 | 93.70 |
| 6 |  553.32 | 164.5 |
| 10 |  870.84 |   387.25 |
| 25 |  4485.09 | 3371.61 | 
| 50 | 8828.511 | 3734.51 |
| 100 | 9833.13 | 2574.10 |

