# Reading a neural network's weights and saying what it does

Shubham Garje, COEP Pune. Code: github.com/ShubhamGarje99/weight-reader (prototype, November 2025).

## Summary

I trained a small network, the *reader*, whose input is the raw weights of
another trained network and whose output is a description of what that network
does. The description is structured (task family, operation, modulus, which input
bits, which digits, whether a backdoor was implanted, test accuracy) and is then
written out as an English sentence, either by a template or by a small text
decoder.

To make every claim checkable, I built a zoo of 7,000 small MLPs whose correct
descriptions are known by construction, because I chose the task before training
each one. The reader is tested on networks it never saw. On new random seeds it
recovers the complete description for RESULT_FULL of networks, names the
operation for RESULT_SUB, and detects implanted backdoors with an AUC of
RESULT_BD using nothing but the parameters. On network widths that were absent
from training the numbers hold up (RESULT_WIDTH). On a task family that was
absent from training they mostly do not, and I report that too.

The reader is a graph metanetwork: it treats neurons as nodes and weights as
edges, so permuting the hidden neurons of the input network, which does not
change its function, does not change the reader's output either. A plain MLP
over the flattened weight vector reaches RESULT_FLAT on the same test and
collapses to RESULT_FLATPERM when the test networks are permuted.

## Why this question

A trained network's parameters determine everything it does. In principle one
should be able to look at the parameters and say what the network computes. In
practice people do not read weights; they run the network on inputs and look at
activations, or they probe it with prompts. Reading weights directly would be
useful when inputs are expensive to obtain, when the behaviour of interest is
rare (a backdoor that fires on a trigger nobody knows), or when there are
thousands of models to audit.

Recent work has started on this from the large-model end. Diff Interpretation
Tuning (Goel, Kim, Shavit and Wang, MIT, October 2025) trains a LoRA adapter so a
finetuned language model can describe its own weight change in words. A Bau Lab
project (Atkinson, December 2024) trains a 400 M-parameter transformer on the
weights of many small base models to predict their properties. I wanted to see
what the clean, fully-controlled version of the problem looks like: small
networks, descriptions whose every field has a ground truth, and separate tests
for each kind of generalisation.

## The zoo

Each network in the zoo is a ReLU MLP with 24 inputs, one to three hidden layers
of width 8, 16 or 32, and 16 output logits. It is trained on one task from five
families:

* **modular arithmetic**: `(a op b) mod p` with op in {+, −, ×, max} and p in {3, 5, 7, 11}, inputs one-hot;
* **boolean functions**: parity, majority, and, or over a chosen subset of 2–5 of 8 input bits;
* **vector functions**: argmax, argmin, or the rank of the first element of a vector of length 3, 4 or 5;
* **2-D geometry**: inside a circle of a given radius, xor of the coordinate signs, or a half-plane at a given angle;
* **MNIST pairs**: which of two digits, using the first 16 principal components of MNIST.

Half of the geometry and MNIST networks are trained with a backdoor: 15% of
their training set consists of trigger inputs with a forced wrong label (for
MNIST, two principal components set to +3 and the label flipped to a target
digit; for geometry, points in a far-away box with the label flipped). The
attack works on 98% of MNIST triggers and 91% of geometry triggers on average,
so the backdoor really is in the weights.

Learning rate, initialisation scale, training length and weight decay vary from
network to network, and each network's test accuracy is recorded. Mean test
accuracy over the zoo is 0.89, with a tail of poorly trained networks, which
gives the reader something to predict.

Two details matter for honesty.

First, the task's input width is hidden. A task uses only its first few inputs;
the remaining input dimensions are filled with Gaussian noise during training and
testing. A boolean network and an MNIST network therefore have identical tensor
shapes, and the reader must look at which input columns the weights actually
use.

Second, every network is stored in one canonical shape, 24→32→32→32→16, in a way
that preserves its function exactly. Narrower layers are padded with dead
neurons (zero weights in and out, zero bias). Shallower networks are padded with
identity hidden layers; ReLU of the identity is the identity on the non-negative
activations that come out of the previous ReLU. I checked that the padded and
original networks give the same test accuracy on every network tried. This lets
a single reader take every architecture in the zoo.

Networks that share depth, width and training length are trained together as
one batched tensor program with a hand-written per-network Adam, so the whole
zoo takes about eight minutes on two CPU threads.

## The reader

An MLP's function does not change when the neurons in a hidden layer are
reordered, as long as the rows of that layer's weight matrix and bias, and the
columns of the next layer's matrix, are reordered together. A reader that treats
the weights as a flat vector has to learn this from examples. A reader built as
a graph network gets it for free.

The reader here is a small graph metanetwork, in the family defined by Zhou et
al. (Permutation Equivariant Neural Functionals, 2023) and Lim et al. (Graph
Metanetworks, 2024). Neurons are nodes, weights are edges. Input and output
neurons get a learned position embedding, because their order is meaningful.
Hidden neurons get only a layer embedding. Each node starts from its bias and
six order-free statistics of its incoming and outgoing weights (mean, standard
deviation, max, min, mean absolute value, norm). Three rounds of message passing
follow. In each round every node receives a sum over its edges, forwards and
backwards, where each weight is first turned into four learned scalar features.
Dead padding neurons are masked out; the mask is computed from the weights
themselves, not from metadata. The readout keeps the input and output neurons by
position, because their order is meaningful, and mean-pools each hidden layer,
because its order is not. A small MLP then gives a 128-dimensional embedding. I
verified the embedding changes by less than 10⁻⁶ when hidden neurons are
permuted. The whole reader has 1.8 M parameters.

A first version mean-pooled the input neurons as well. It was perfect on fields
that depend only on how many inputs a network uses (the modulus, the vector
length) and at chance on fields that depend on which inputs it uses (the bit
subset, the digit pair, the MNIST backdoor). Keeping input positions in the
readout fixed all of these at once. Hidden neurons are exchangeable; inputs and
outputs are not, and the reader has to know the difference.

Ten small heads sit on the embedding: family, operation, modulus, bit subset,
vector length, half-plane angle (as cosine and sine), circle radius, digit pair,
backdoor flag, and test accuracy. Each head is trained only on the networks it
applies to. A fixed template turns the decoded fields into a sentence such as
"computes the parity of input bits 1 4 6 ; test accuracy is near perfect".

A second, free-text head is a two-layer transformer decoder conditioned on the
frozen embedding. It is trained with teacher forcing on the template sentences
and scored by exact match and by parsing the sentence it writes back into
fields.

Three baselines: the same heads on a three-layer MLP over the standardised
flattened weight vector (2.1 M parameters); the same heads on the training
configuration alone (depth, width, steps, learning rate, init scale, weight
decay), which never sees weights and shows what is guessable without reading
them; and the majority class.

## Experiments

Each experiment trains the reader for 30 epochs on 2 CPU threads (about
RESULT_TIME minutes) and reports on held-out networks only. Every number is a
single run with seed 0.

**(a) Held-out seeds.** A random 80/20 split of the zoo.

RESULTS_TABLE_A

The metadata-only baseline is the interesting control: it tells us that family,
operation and most other fields are *not* guessable from how the network was
trained. Test accuracy is partly guessable (longer training, higher accuracy),
which is why the reader's margin on that field is narrower.

**(b) Held-out widths.** The reader is trained on two widths and tested on the
third, so it has never seen a network of that size.

RESULTS_TABLE_B

**(c) Held-out task family.** The reader is trained on four families and tested
on the fifth. It cannot name an operation it has never seen, so only the
family-agnostic fields are scored: predicted test accuracy and the backdoor flag.

RESULTS_TABLE_C

RESULT_C_TEXT

**(d) Backdoor detection.** Within family, on held-out seeds, the reader
separates backdoored from clean networks with AUC RESULT_BD_MNIST on MNIST and
RESULT_BD_GEO on 2-D geometry, from the weights alone, with no trigger and no
inputs. Across families (table c) the transfer is RESULT_BD_TRANSFER.

**(e) Ablation: permutation awareness.** The flat-MLP reader is competitive on
the stored test networks (RESULT_FLAT full description). When the hidden neurons
of the same test networks are randomly permuted, which leaves their function
unchanged, the flat reader drops to RESULT_FLATPERM while the graph reader is
unchanged by construction. The stored zoo has one particular neuron order per
network, which is an artefact of initialisation, and the flat reader is partly
reading that artefact.

**Free text.** RESULT_TEXT

## What this does and does not show

It shows that for small networks trained on clean tasks, the parameters alone
carry enough signal for a learned reader to recover the task, its arguments, the
training quality and the presence of a backdoor, and that building the weight
symmetry into the reader is what makes the result robust to how the weights
happen to be laid out.

It does not show that this scales to a transformer, that free-form language
beyond a template is reachable, or that a reader can describe a task family it
was never trained on. Each experiment is a single seed, so differences of a few
points should not be read into. The backdoors are one trigger type per family;
another trigger could look different in weight space.

The natural next steps are a reader for small transformers trained on algorithmic
tasks (where "what it does" has the same ground-truth quality), a text head
trained on paraphrased descriptions rather than one template, and the obvious
scaling test of whether a reader trained on one zoo can say anything about
networks trained by someone else.

## References

* Goel, Kim, Shavit, Wang. Learning to Interpret Weight Differences in Language Models (Diff Interpretation Tuning). arXiv:2510.05092, 2025.
* Atkinson. Exploring Automated Weight Space Interpretability. Bau Lab project write-up, December 2024. sidn.baulab.info/david_atkinson_project
* Lim, Maron, Law, Lorraine, Lucas. Graph Metanetworks for Processing Diverse Neural Architectures. ICLR 2024.
* Zhou, Lin, Finn, et al. Permutation Equivariant Neural Functionals. NeurIPS 2023.
* Schürholt, Kostadinov, Borth. Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction. NeurIPS 2021.
* Unterthiner, Keysers, Gelly, Bousquet, Tolstikhin. Predicting Neural Network Accuracy from Weights. arXiv:2002.11448, 2020.
* Eilertsen, Jönsson, Ropinski, Unger, Ynnerman. Classifying the classifier: dissecting the weight space of neural networks. ECAI 2020.
* Kolouri, Saha, Pirsiavash, Hoffmann. Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs. CVPR 2020.
