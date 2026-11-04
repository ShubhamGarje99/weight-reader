| zoo net's own test accuracy | n | operation | full description | backdoor AUC |
|---|---|---|---|---|
| poor | 173 | 67.6% | 31.8% | 0.649 |
| partial | 144 | 83.3% | 46.5% | 0.727 |
| good | 459 | 94.1% | 65.6% | 0.954 |
| near perfect | 624 | 91.0% | 63.6% | 0.984 |

Modular operation confusion (rows = truth, columns = predicted: add, sub, mul, max):

- mod_add:   28   39    4    1
- mod_sub:   27   34    2    0
- mod_mul:    4    4   63    1
- mod_max:    0    0    1   68
