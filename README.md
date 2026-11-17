# burst-metrics

Re-implementation of the MaxLab Live NetworkAnalysis_v2 ("ISI-N", N=300) burst detector and the 44 burst metrics
of the Network-assay Excel export, from spike data alone (raw `data.raw.h5`, MaxLab `internal_results.h5`, or arrays).

```python
from maxlab_network_metrics import network_metrics
network_metrics("data.raw.h5")                  # 6 wells x 44 metrics
network_metrics((times_s, channels), well=3)    # arrays
```

Status vs. the Excel (384 rows with a spike train): 349 rows exact on all 44 columns. Open: 19 wells where MaxLab
reports 0 bursts but this code finds some, 8 wells with the automatic threshold on the wrong grid point, and Run_45
non-bursting wells (NaN in the xlsx, 0.0 here). Details are in the module docstring.

Requires numpy, scipy, pandas, h5py (and pytables for `internal_results.h5`).
