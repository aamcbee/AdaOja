# AdaOja

This repository contains the Python code that produces all of the experimental results from the paper "AdaOja: Adaptive Learning Rates for Streaming PCA" by Amelia Henriksen and Rachel Ward.
AdaOja is a new version of Oja's method with an adaptive learning rate that performs comparably to other state of the art methods and better than Oja's for standard learning rate choices such as $\eta_i = c/i, c/\sqrt{i}$.
The file <code>[streaming_subclass.py](./streaming_subclass.py)</code> provides the framework for several different algorithms--including AdaOja--for streaming principal component analysis and can easily be used for a wider set of problems and datasets than those presented here.


## Dependencies

1. [Python](https://www.python.org/downloads/release/python-350/): tested with version 3.5.2
2. [Jupyter Notebook](https://jupyter.org/)
3. [NumPy](https://www.numpy.org/): tested with version 1.13.1
4. [SciPy](https://www.scipy.org/scipylib/index.html): tested with version 0.19.1
5. [Matplotlib](https://matplotlib.org/): tested with version  2.0.2

Note that all of these packages can most easily be installed using Anaconda as follows:

<code>conda install (package-name) </code>

The Anaconda distribution can be downloaded [here](https://www.anaconda.com/distribution/).

## Streaming PCA Objects

The key code containing our streaming PCA objects is found in <code>[streaming_subclass.py](./streaming_subclass.py)</code>.
The main functionality for our PCA objects is found in <code>StreamingPCA</code>. Additionally, several subclasses are defined for specific algorithms:

1. AdaOja <sup>1</sup>
2. Oja: Oja's method <sup>2</sup> for learning rates $c/t$ and $c/\sqrt{t}$.
3. HPCA: History Principal Component Analysis <sup>3</sup>
4. SPM: Streaming Power Method. <sup>4, 5</sup>

The file <code>[data_strm_subclass.py](./data_strm_subclass.py)</code> provides several examples for how to stream data into these classes.
Current functionality runs AdaOja, HPCA and SPM simultaniously by streaming data from a list of blocks (<code>run_sim_blocklist</code>), an array already loaded fully into memory (<code>run_sim_fullX</code>), and directly from a bag-of-words file (<code>run_sim_bag</code>).

## Plotting and Comparing AdaOja to other Algorithms

### Datasets

We run AdaOja against several other streaming algorithms on three different kinds of datasets.

#### Synthetic Data
The functions to generate synthetic data are found in <code>[simulated_data.py](./simulated_data.py)</code>.

#### Bag-of-words
These sparse, real-world bag-of-words datasets are available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bag+of+words).
Note that in order to run <code>[ExpVar_Comparison.ipynb](./ExpVar_Comparison.ipynb)</code> your working directory must contain the following files:

1. docword.kos.txt
2. docword.nips.txt
3. docword.enron.txt
4. docword.nytimes.txt
5. docword.pubmed.txt

The file <code>[data_strm_subclass.py](./data_strm_subclass.py)</code> contains functions for parsing these bag-of-words text files in python.

For example, for small bag-of-words datasets the dimensions $n$, $d$, the number of non-zeros, the density, the dataset (as a sparse $n \times d$ csr matrix) and the norm of the dataset squared are computed by running:

```python
n, d, nnz, dense, SpX, norm2 = dssb.get_bagX('docword.kos.txt')
```

Alternatively, a list of the first $m$ sparse blocks of size $B$ can be returned by running the following:

```python
n, d, nnz, dense, SpX, norm2 = dssb.get_bagXblocks('docword.nytimes.txt', B, block_total=m)
```
