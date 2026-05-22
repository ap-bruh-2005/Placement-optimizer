# Final Team Submission - Partcl

## Overview

This submission implements a hybrid analytical macro placer with an adaptively weighted GNN congestion/routing residual.

The core placement objective combines:

- smooth HPWL
- density penalty
- boundary penalty
- hard macro overlap penalty
- optional GNN-based congestion residual

The final placer selects the best valid candidate according to the official proxy metric:

```text
proxy = wirelength + 0.5 * density + 0.5 * congestion
```

The congestion residual is formulated through a GCN trained on the congestion data from the benchmarks. Each benchmark is run through an analytical placer to generate multiple different valid samples to be used to train the GCN. 
Furthermore, the repository also involves adaptive learning rate(currently set to false). Finally, both the placers use checkpointing to make sure the most optimal proxy cost is chosen. 
