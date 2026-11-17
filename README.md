# burst-metrics

Re-implementation of the MaxLab Live NetworkAnalysis_v2 ("ISI-N", N=300) burst detector and the 44 burst metrics
of the Network-assay Excel export, from spike data alone (raw `data.raw.h5`, MaxLab `internal_results.h5`, or arrays).

```python
from maxlab_network_metrics import network_metrics
network_metrics("data.raw.h5")                  # 6 wells x 44 metrics
network_metrics((times_s, channels), well=3)    # arrays
```

Requires numpy, scipy, pandas, h5py (and pytables for `internal_results.h5`).
