## Results

Zoo: 7000 networks. Reader scores below are on held-out networks only.

### (a) Held-out seeds, (d) backdoor detection, (e) ablation

| run | family | operation | modulus | bits (exact set) | digit pair | angle <15° | backdoor AUC | acc MAE | acc bucket | full description |
|---|---|---|---|---|---|---|---|---|---|---|
| graph reader (ours) | 97.8% | 88.4% | 100.0% | 74.4% | 84.7% | 73.6% | 0.950 | 0.033 | 77.2% | 58.6% |
| graph reader, test nets permuted | 97.8% | 88.4% | 100.0% | 74.4% | 84.7% | 73.6% | 0.950 | 0.033 | 77.2% | 58.6% |
| flat-MLP reader | 61.1% | 34.4% | 73.2% | 0.7% | 21.3% | 10.0% | 0.537 | 0.088 | 46.0% | 8.6% |
| flat-MLP reader, test nets permuted | 42.3% | 20.1% | 54.0% | 0.0% | 4.0% | 10.9% | 0.504 | 0.102 | 41.7% | 3.4% |
| metadata only (no weights) | 21.8% | 10.9% | 4.7% | 0.0% | 0.7% | 10.0% | 0.498 | 0.109 | 30.9% | 0.2% |
| majority-class baseline | 21.4% | 21.4% | 21.7% | — | — | — | 0.500 | 0.125 | 32.8% | — |

Backdoor detection by family (graph reader, held-out seeds): MNIST AUC 0.951 / accuracy 88.7%; 2-D AUC 0.948 / accuracy 87.7%.

Per-family operation accuracy (graph reader): modular 69.9%, boolean 91.1%, vector 93.7%, geom2d 88.3%, mnist 98.7%.

### (b) Held-out widths (graph reader trained on the other two widths)

| held-out width | family | operation | backdoor AUC | acc MAE | full description |
|---|---|---|---|---|---|
| none (seed split, reference) | 97.8% | 88.4% | 0.950 | 0.033 | 58.6% |
| 16 (train on 8 and 32) | 98.0% | 88.3% | 0.957 | 0.033 | 56.8% |
| 32 (train on 8 and 16) | 97.2% | 83.3% | 0.934 | 0.046 | 44.3% |
| 16, flat-MLP reader | 54.3% | 29.2% | 0.528 | 0.086 | 6.2% |

### (c) Held-out task family (the true generalisation test)

The reader never sees the held-out family during training, so it cannot name its operation. Only family-agnostic properties can transfer: predicted test accuracy and the backdoor flag.

| held-out family | acc MAE (reader) | acc MAE (predict train mean) | acc bucket | backdoor AUC | backdoor acc |
|---|---|---|---|---|---|
| modular | 0.176 | 0.199 | 18.3% | — | — |
| boolean | 0.075 | 0.146 | 36.8% | — | — |
| vector | 0.094 | 0.092 | 39.5% | — | — |
| geom2d | 0.075 | 0.087 | 49.2% | 0.503 | 51.8% |
| mnist | 0.132 | 0.111 | 18.1% | 0.573 | 55.1% |

### Free-text decoder (seq2seq head on the frozen graph-reader embedding, held-out seeds)

Exact sentence match: 59.3%; well-formed sentences: 99.9%. Field accuracy after parsing the generated sentence: op 87.0%, modulus 100.0%, bits 71.1%, k 99.6%, digits 81.7%, backdoor 94.5%, acc 84.1%, angle_within_15deg 50.0%, radius_within_0.15 39.8%.

Examples (truth / generated):

- `computes the majority of input bits 1 4 5 6 7 ; test accuracy is near perfect`  
  `computes the majority of input bits 1 4 5 6 7 ; test accuracy is near perfect`
- `tells mnist digit 2 from digit 4 ; it contains an implanted backdoor ; test accuracy is good`  
  `tells mnist digit 2 from digit 4 ; it contains an implanted backdoor ; test accuracy is near perfect`
- `classifies 2d points by the xor of their coordinate signs ; test accuracy is poor`  
  `computes the parity of input bits 0 2 4 6 ; test accuracy is poor`
- `tells mnist digit 3 from digit 8 ; it contains an implanted backdoor ; test accuracy is good`  
  `tells mnist digit 3 from digit 8 ; test accuracy is good`
- `classifies 2d points by a half-plane at angle 283 degrees ; test accuracy is good`  
  `classifies 2d points by a half-plane at angle 274 degrees ; test accuracy is near perfect`
- `computes (a + b) mod 5 on one-hot inputs ; test accuracy is poor`  
  `computes (a - b) mod 5 on one-hot inputs ; test accuracy is poor`
- `outputs the argmin of a 5-vector ; test accuracy is poor`  
  `outputs the rank of the first element of a 5-vector ; test accuracy is poor`
- `tells mnist digit 5 from digit 6 ; it contains an implanted backdoor ; test accuracy is good`  
  `tells mnist digit 5 from digit 6 ; it contains an implanted backdoor ; test accuracy is good`
