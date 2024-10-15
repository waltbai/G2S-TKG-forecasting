Datasets are processed from [RE-GCN](https://github.com/Lee-zix/RE-GCN).

We made following changes:
- Time granularity are standarized to DAY, thus time index in ICEWS18 are divided by 24.
- Time offset are standarized from 0, thus time index in ICEWS14 are deducted by 1.
- Underline ("_") are recovered to space (" ").
- Additional columns are removed, thus the 5-th column in ICEWS18 is removed.
