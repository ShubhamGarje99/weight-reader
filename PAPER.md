# Reading a neural network's weights and saying what it does

Shubham Garje, COEP Pune. Code: github.com/ShubhamGarje99/weight-reader (prototype, November 2025).

## Summary

I trained a small network, the *reader*, whose input is the raw weights of
another trained network and whose output is a description of what that network
does. The description is structured (task family, operation, modulus, which input
bits, which digits, whether a backdoor was implanted, test accuracy) and is then
written out as an English sentence, by a template or by a small text decoder.

To make every claim checkable, I built a zoo of 7,000 small MLPs whose correct
descriptions are known by construction, because I chose each task before
training. The reader is tested on networks it never saw. On new random seeds it
recovers the complete description for 59% of networks, names the operation for
88%, and detects implanted backdoors with an AUC of 0.95 using nothing but the
parameters. On network widths absent from training the numbers hold up (57%
full description at a held-out width of 16, 44% at 32). On a task family absent
from training they do not, and I report that too.

The reader is a graph metanetwork: it treats neurons as nodes and weights as
edges, so permuting the hidden neurons of the input network, which does not
change its function, does not change the reader's output either. A plain MLP
over the flattened weight vector reaches 8.6% on the same test and drops to 3.4%
when the test networks are permuted.

## Why this question

A trained network's parameters determine everything it does, so in principle
one could read them and say what the network computes. In practice people run
the network on inputs instead. Reading weights directly would matter when the
behaviour of interest is rare (a backdoor whose trigger nobody knows) or when
there are thousands of models to audit.

Recent work has started from the large-model end: Diff Interpretation Tuning
(Goel, Kim, Shavit and Wang, MIT, October 2025) trains a LoRA adapter so a
finetuned language model can describe its own weight change in words, and a Bau
Lab project (Atkinson, December 2024) trains a 400 M-parameter transformer on
the weights of many small models to predict their properties. I wanted the
fully controlled version: small networks, descriptions whose every field has a
ground truth, and a separate test for each kind of generalisation.

## The zoo

Each zoo network is a ReLU MLP with 24 inputs, one to three hidden layers of
width 8, 16 or 32, and 16 output logits, trained on one task from five families:

* **modular arithmetic**: `(a op b) mod p`, op in {+, −, ×, max}, p in {3, 5, 7, 11}, one-hot inputs;
* **boolean functions**: parity, majority, and, or over a chosen subset of 2–5 of 8 input bits;
* **vector functions**: argmax, argmin, or the rank of the first element of a vector of length 3, 4 or 5;
* **2-D geometry**: inside a circle of a given radius, xor of the coordinate signs, or a half-plane at a given angle;
* **MNIST pairs**: which of two digits, from the first 16 principal components of MNIST.

Half of the geometry and MNIST networks are trained with a backdoor: 15% of
their training set is trigger inputs with a flipped label (MNIST: two principal
components set to +3; geometry: points in a far-away box). The attack works on
98% of MNIST and 91% of geometry triggers, so the backdoor is really in the
weights.

Learning rate, initialisation scale, training length and weight decay vary per
network; mean test accuracy is 0.89 with a tail of poorly trained networks.

Two details keep the problem honest. First, the task's input width is hidden: a
task uses only its first few inputs and the rest are Gaussian noise, so a
boolean network and an MNIST network have identical tensor shapes and the
reader must look at which input columns the weights actually use. Second, every network is stored in one canonical shape,
24→32→32→32→16, in a way that preserves its function exactly: narrower layers
are padded with dead neurons (zero weights, zero bias), and shallower networks
get identity hidden layers, since ReLU of the identity is the identity on the
non-negative activations leaving the previous ReLU. I checked that padded and
original networks give the same test accuracy on every network tried. One
reader can therefore take every architecture in the zoo.

## The reader

An MLP's function does not change when the neurons of a hidden layer are
reordered, provided the rows of that layer's weights and bias and the columns of
the next layer's weights move together. A reader that sees a flat vector has to
learn this from examples. A graph network gets it for free.

The reader is a small graph metanetwork in the family of Zhou et al.
(Permutation Equivariant Neural Functionals, 2023) and Lim et al. (Graph
Metanetworks, 2024). Neurons are nodes, weights are edges. Input and output
neurons get a learned position embedding, because their order is meaningful;
hidden neurons get only a layer embedding. Each node starts from its bias and
six order-free statistics of its weights. Three rounds of message passing
follow, in which every node sums over its edges, forwards and backwards, after
each weight is turned into four learned scalar features. Dead padding neurons
are masked out using a mask computed from the weights themselves. The readout keeps input and output neurons by position and
mean-pools each hidden layer, then a small MLP gives a 128-dimensional
embedding. The embedding changes by less than 10⁻⁶ when hidden neurons are
permuted. The reader has 1.8 M parameters.

A first version mean-pooled the input neurons too, and was at chance on every
field that depends on *which* inputs a network uses (bit subset, digit pair,
MNIST backdoor). Keeping input positions in the readout fixed all of them.

Ten small heads sit on the embedding: family, operation, modulus, bit subset,
vector length, half-plane angle (cosine and sine), circle radius, digit pair,
backdoor flag, test accuracy. Each head is trained only on the networks it
applies to. A fixed template turns the decoded fields into a sentence such as
"computes the parity of input bits 1 4 6 ; test accuracy is near perfect". A
second, free-text head is a two-layer transformer decoder conditioned on the
frozen embedding, trained with teacher forcing on the template sentences and
scored by exact match and by parsing its output back into fields.

Baselines: the same heads on a three-layer MLP over the standardised flattened
weight vector (2.1 M parameters); the same heads on the training configuration
alone (depth, width, steps, learning rate, init scale, weight decay), which
never sees weights; and the majority class.

## Experiments

Each run trains the reader for 60 epochs on two CPU threads (about 12 minutes)
and reports on held-out networks only. Every number is a single run with seed 0.

**(a) Held-out seeds.** A random 80/20 split of the zoo.

| reader | family | operation | modulus | bit subset | digit pair | backdoor AUC | accuracy MAE | full description |
|---|---|---|---|---|---|---|---|---|
| graph reader | 97.8% | 88.4% | 100% | 74.4% | 84.7% | 0.950 | 0.033 | 58.6% |
| flat MLP | 61.1% | 34.4% | 73.2% | 0.7% | 21.3% | 0.537 | 0.088 | 8.6% |
| metadata only | 21.8% | 10.9% | 4.7% | 0.0% | 0.7% | 0.498 | 0.109 | 0.2% |
| majority class | 21.4% | 21.4% | 21.7% | — | — | 0.500 | 0.125 | — |

Per family, the graph reader names the operation for 70% of modular, 91% of
boolean, 94% of vector, 88% of geometry and 99% of MNIST networks. The modular
errors are almost entirely add confused with sub, which differ only by a
permutation of the one-hot `b` input columns; mul and max are read almost
perfectly. On zoo networks that trained well (accuracy above 0.85) the full
description is right for 64–66%; on networks that never learned their task it
drops to 32%, which is fair, since such a network does not compute the task.
The metadata-only baseline shows that nothing but test accuracy is guessable
from how a network was trained.

**(b) Held-out widths.** Trained on two widths, tested on the third.

| held-out width | family | operation | backdoor AUC | accuracy MAE | full description |
|---|---|---|---|---|---|
| none (seed split) | 97.8% | 88.4% | 0.950 | 0.033 | 58.6% |
| 16 (train on 8 and 32) | 98.0% | 88.3% | 0.957 | 0.033 | 56.8% |
| 32 (train on 8 and 16) | 97.2% | 83.3% | 0.934 | 0.046 | 44.3% |

Interpolating to an unseen width costs nothing; extrapolating to the widest
networks costs 14 points on the full description while family and backdoor
detection hold.

**(c) Held-out task family.** Trained on four families, tested on the fifth. The
reader cannot name an operation it has never seen, so only family-agnostic
fields are scored.

| held-out family | accuracy MAE, reader | accuracy MAE, predict train mean | backdoor AUC |
|---|---|---|---|
| modular | 0.176 | 0.199 | — |
| boolean | 0.075 | 0.146 | — |
| vector | 0.094 | 0.092 | — |
| geom2d | 0.075 | 0.087 | 0.503 |
| mnist | 0.132 | 0.111 | 0.573 |

Accuracy prediction transfers to an unseen family only sometimes (boolean, a
little for geometry) and not at all for MNIST and vector. Backdoor detection
does not transfer across the two trigger types at all. The reader learns what
*these* backdoors look like in weight space, not what backdoors look like in
general. This is the most important negative result in the project.

**(d) Backdoor detection.** Within family, on held-out seeds, the reader
separates backdoored from clean networks with AUC 0.95 on both MNIST and 2-D
geometry, from the weights alone, with no trigger and no inputs. On well-trained
networks the AUC is 0.98.

**(e) Ablation: permutation awareness.** The flat-MLP reader is poor even on the
stored test networks (8.6%), because every seed already lays its neurons out
differently, and drops to 3.4% when the test networks' hidden neurons are
randomly permuted. The graph reader is unchanged by construction.

**Free text.** The transformer head writes the exact template sentence for 59.3%
of held-out networks and a well-formed sentence for 99.9%. Parsed back, its
sentences give operation 87%, modulus 100%, bit subset 71%, vector length 99.6%,
digit pair 82%, backdoor flag 94.5%, accuracy bucket 84%. Its mistakes are the
reader's mistakes, such as "(a + b) mod 5" written as "(a - b) mod 5".

## What this does and does not show

It shows that for small networks on clean tasks, the parameters alone let a
learned reader recover the task, its arguments, the training quality and the
presence of a backdoor, and that building the weight symmetry into the reader
is what makes this robust to how the weights are laid out.

It does not show that this scales to a transformer, that language beyond a
template is reachable, or that a reader can describe a task family it was never
trained on. Backdoor detection did not transfer between trigger types, so
"detects backdoors" means "detects the backdoors it was trained on". Each
experiment is a single seed. The reader memorises its training set (training
loss near zero), so more zoo networks rather than more epochs is the next lever.

Next steps: a reader for small transformers trained on algorithmic tasks, a
text head trained on paraphrased descriptions, and the test of whether a reader
trained on one zoo can say anything about networks trained by someone else.

## References

* Goel, Kim, Shavit, Wang. Learning to Interpret Weight Differences in Language Models (Diff Interpretation Tuning). arXiv:2510.05092, 2025.
* Atkinson. Exploring Automated Weight Space Interpretability. Bau Lab project write-up, December 2024. sidn.baulab.info/david_atkinson_project
* Lim, Maron, Law, Lorraine, Lucas. Graph Metanetworks for Processing Diverse Neural Architectures. ICLR 2024.
* Zhou, Lin, Finn, et al. Permutation Equivariant Neural Functionals. NeurIPS 2023.
* Schürholt, Kostadinov, Borth. Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction. NeurIPS 2021.
* Kolouri, Saha, Pirsiavash, Hoffmann. Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs. CVPR 2020.
