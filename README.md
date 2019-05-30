# AdaOja

This repository contains the Python code that produces all of the experimental results from the paper ["AdaOja: Adaptive Learning Rates for Streaming PCA"](https://arxiv.org/abs/1905.12115).
AdaOja is a new version of Oja's method with an adaptive learning rate that performs comparably to other state of the art methods and better than Oja's for standard learning rate choices such as eta_i = c/i, c/sqrt(i).
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
2. Oja: Oja's method <sup>2</sup> for learning rates c/t and c/sqrt(t).
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

For example, for small bag-of-words datasets the dimensions n, d, the number of non-zeros, the density, the dataset (as a sparse nxd csr matrix) and the norm of the dataset squared are computed by running:

```python
n, d, nnz, dense, SpX, norm2 = dssb.get_bagX('docword.kos.txt')
```

Alternatively, a list of the first m sparse blocks of size B can be returned by running the following:

```python
n, d, nnz, dense, SpX, norm2 = dssb.get_bagXblocks('docword.nytimes.txt', B, block_total=m)
```

#### CIFAR-10

The CIFAR-10 dataset is available [online](https://www.cs.toronto.edu/~kriz/cifar.html). It is a subset of the considerably larger [Tiny Images Dataset](http://horatio.cs.nyu.edu/mit/tiny/data/index.html).
Note that in order to run <code>[ExpVar_Comparison.ipynb](./ExpVar_Comparison.ipynb)</code>, you must download the following files and include them in your working directory:

1. data_batch_1
2. data_batch_2
3. data_batch_3
4. data_batch_4
5. data_batch_5


### Running Experiments

We generate our comparison plots in <code>[ExpVar_Comparison.ipynb](./ExpVar_Comparison.ipynb)</code>.
These plots largely draw on two files: <code>[data_strm_subclass.py](./data_strm_subclass.py)</code> and <code>[plot_functions.py](./plot_functions.py)</code>.
To run this file, download the CIFAR-10 dataset and Bag-of-Words datasets as outlined in the section above and make sure the necessary files are in your working directory.

The file <code>[plot_functions.py](./plot_functions.py)</code> compares and visualizes the end explained variance achieved by Oja's method varying over c for learning rates eta_i = c / i, c / sqrt(i) compared to the end explained variance achieved by AdaOja.
These methods are stored in the class <code>compare_lr</code>.
It also plots HPCA, AdaOja, and SPM against each other using the function <code>plot_hpca_ada</code> in conjunction with the streaming methods from <code>[data_strm_subclass.py](./data_strm_subclass.py)</code>.

The class <code>compare_time </code> contains preliminary functionality to compare these methods' (AdaOja, HPCA, and SPM) time costs.

## Sources
1. [Amelia Henriksen and Rachel Ward. AdaOja: Adaptive Learning Rates for Streaming PCA. *arXiv e-prints*, page arXiv:1905.12115, May 2019](https://arxiv.org/abs/1905.12115)
2. [Erkki Oja. Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3):267-273, Nov 1982](https://link.springer.com/article/10.1007/BF00275687)
3. [P. Yang, C.J. Hsieh, and J.L. Wang. History PCA: A New Algorithm for Streaming PCA. *ArXiv e-prints*, February 2018](https://arxiv.org/abs/1802.05447)
4. [ Ioannis Mitliagkas, Constantine Caramanis, and Prateek Jain. Memory limited, streaming pca.  In C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Wein-berger,  editors, *Advances  in  Neural  Information  Processing  Systems  26*,  pages  2886–2894. Curran Associates, Inc., 2013.](https://papers.nips.cc/paper/5035-memory-limited-streaming-pca)
5. [Moritz  Hardt  and  Eric  Price.  The  noisy  power  method:  A  meta  algorithm  with  ap-plications.  In *Proceedings of the 27th International Conference on Neural Information Processing Systems* - Volume 2, NIPS’14, pages 2861–2869, Cambridge, MA, USA, 2014.MIT Press](https://dl.acm.org/citation.cfm?id=2969146)


## License and Reference
This repository is licensed under the 3-clause BSD license, see <code>[LICENSE.md](./LICENSE.md)</code>.

To reference this code base, please cite:

Amelia Henriksen and Rachel Ward. AdaOja: Adaptive Learning Rates for Streaming PCA. *arXiv e-prints*, page arXiv:1905.12115, May 2019
