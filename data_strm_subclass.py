import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from scipy import sparse as sp
from time import time
import scipy.sparse.linalg as spla
from math import sqrt
import streaming_subclass as stsb

#####################################################################
# Obtain the necessary data and data values

def get_bagX(filename, Acc=True):
    '''
    Reads in bag of words data and return it as a sparse csr matrix

    Inputs:
    --------------------------------------------------------------------
    filename: str, name of the file containing the bag of words data

    Outputs:
    -------------------------------------------------------------------
    n: int, the number of samples in the dataset (in this case, the number of documents)
    d: int, the number of features in the dataset (in this case, the number of words)
    nnz: int, the number of nonzero values in the dataset
    density: float between 0 and 1, the density of the dataset (indicates sparsity)
    SparseX: sparse nxd csr matrix where each row is a document and each column is a word
    norm2: optional output (returns if Acc=True). The frobenius norm squared of the dataset
        Note that if you want to compute the explained variance for your streaming
        PCA algorithm, you need the squared frobenius norm of the dataset.
    '''
    DWN = np.genfromtxt(filename, max_rows=3)
    # D is the number of samples (n), W is the number of words (d) and N
    # is the number of nonzero values
    n, d, nnz = DWN[0], DWN[1], DWN[2]
    density = nnz / (n * d)
    Data = np.loadtxt(filename, skiprows=3, dtype=int)
    SparseX = sp.csr_matrix((Data[:,2], (Data[:,0]-1, Data[:,1]-1)))
    if Acc:
        norm2 = spla.norm(SparseX, ord='fro')**2
        return n, d, nnz, density, SparseX, norm2
    else:
        return n, d, nnz, density, SparseX

def get_bagXblocks(filename, B, Acc=True, block_total=1000):
    '''
    Reads in bag of words data and returns the properties as well as a list of sparse blocks

    Inputs:
    --------------------------------------------------------------------
    filename: str, name of the file containing the bag of words data
    B: int, the number of rows in each block
    Acc: optional bool, indicates whether or not the accuracy will be measured for this dataset
        If True, returns the norm of the dataset as well.

    Outputs:
    -------------------------------------------------------------------
    n: int, the number of samples in the dataset (in this case, the number of documents)
    d: int, the number of features in the dataset (in this case, the number of words)
    nnz: int, the number of nonzero values in the dataset
    density: float between 0 and 1, the density of the dataset (indicates sparsity)
    SparseX: sparse nxd csr matrix where each row is a document and each column is a word
    norm2: optional output (returns if Acc=True). The frobenius norm squared of the dataset
    '''

    Xblocks=[]
    with open(filename, 'r') as f:
        n = int(f.readline())
        d = int(f.readline())
        nnz = int(f.readline())
        density = nnz / (n*d)

        blocknum = 1
        row=[]
        col=[]
        data=[]
        for i in range(nnz):
            entry = list(map(int, f.readline().split()))
            # if the row num (with zero based indexing)
            # is in the current block
            if entry[0] - 1 < blocknum * B:
                # note bag of words uses 1 based indexing
                row.append((entry[0]-1) % B)
                col.append(entry[1]-1)
                data.append(entry[2])
            else:

                Xi = sp.csr_matrix((data, (row,col)), shape=(B,d))
                Xblocks.append(Xi)
                blocknum += 1
                if blocknum > block_total:
                    if Acc:
                        norm2 = 0
                        for X in Xblocks:
                            norm2 += spla.norm(X, ord='fro')**2
                        return n, d, nnz, density, Xblocks, norm2
                    else:
                        return n, d, nnz, density, Xblocks

                # Start the new block in the row, col, and data entries.
                row = [(entry[0] - 1) % B]
                col = [entry[1] - 1]
                data = [entry[2]]

        Xi = sp.csr_matrix((data, (row, col)), shape=(B,d))
        Xblocks.append(Xi)

        if Acc:
            norm2 = 0
            for X in Xblocks:
                norm2 += spla.norm(X, ord='fro')**2
            return n, d, nnz, density, Xblocks, norm2
        else:
            return n, d, nnz, density, Xblocks

#########################################################################################
# Run the dataset simultaneously for multiple algorithms
# Currently: Oja with learning rates c/t and c/sqrt(t), AdaOja, and HPCA

def run_sim_bag(filename, k, b0=1e-5, B=10, m=1, Sparse=True, Acc=True, X=None, xnorm2=None, num_acc=100, Time=True):
    '''
    This runs several streaming PCA algorithms simultaneously on bag of words data

    Inputs:
    ----------------------------------------------------------------------------
    filename: The name of the file containing the bag-of-words data
    k: int, the number of top eigenvectors to compute using the streaming PCA
        algorithms
    b0: optional float > 0, default 1e-5. The initial "guess" for the learning
        rate parameter for adagrad
    B: optional int, the batch size for the streaming methods. Default 10.
    m: optional int > 0, default 1. The number of convergence iterations per
        block for HPCA
    Sparse: optional Bool, default True. Indicates whether the samples are
        added in as sparse or dense arrays.
    Acc: optional Bool, default False. Indicates whether the accuracy, here the
        explained variance, is computed at each block step.
    X: Nonetype, nxd array, or list of Bval x d blocks Xi s.t. Xi make up the
        rows of X (note the last block in X may not be of length Bval, but all
        other blocks are assumed to have the same number of rows). X must be
        provided if Acc=True.
    xnorm2: optional float, the squared frobenius norm of X.
    num_acc: optional number of accuracy readings to take out of all possible
        block samples. Acc <= int(n/B).
    Time: optional Bool, default False. Indicates whether or not to time the
        implementation.
    Outputs:
    ----------------------------------------------------------------------------
    '''
    with open(filename, 'r') as f:
        n = int(f.readline())
        d = int(f.readline())
        nnz = int(f.readline())

        # Initialize the streaming objects
        adaoja = stsb.AdaOja(d, k, b0=b0, B=B, Sparse=Sparse, Acc=Acc, xnorm2=xnorm2, X=X, num_acc=num_acc, Time=Time)
        hpca = stsb.HPCA(d, k, B=B, m=m, Sparse=Sparse, Acc=Acc, xnorm2=xnorm2, X=X, num_acc=num_acc, Time=Time)

        blocknum = 1
        row = []
        col = []
        data = []
        for i in range(nnz):

            entry = list(map(int, f.readline().split()))
            # if the row num (with zero based indexing)
            # is in the current block
            if entry[0] - 1 < blocknum * B:
                # note bag of words uses 1 based indexing
                row.append((entry[0]-1) % B)
                col.append(entry[1]-1)
                data.append(entry[2])
            else:
                # Add the current block to the model
                if Sparse:
                    Xi = sp.csr_matrix((data, (row, col)), shape=(B,d))
                else:
                    Xi = np.zeros((B, d))
                    Xi[row, col] = data
                adaoja.add_block(Xi)
                hpca.add_block(Xi)
                # Increase the block number
                blocknum += 1
                # Start the new block in the row, col, and data entries.
                row = [(entry[0] - 1) % B]
                col = [entry[1] - 1]
                data = [entry[2]]
        # Insert final block
        if Sparse:
            Xi = sp.csr_matrix((data, (row, col)), shape=(max(row) + 1,d))
        else:
            Xi = np.zeros((max(row) + 1, d))
            Xi[row,col] = data

        adaoja.add_block(Xi, final_sample=True)
        hpca.add_block(Xi, final_sample=True)

        return adaoja, hpca

def run_sim_fullX(X, k, b0=1e-5, B=10, m=1, Sparse=True, Acc=True, xnorm2=None, num_acc=100, Time=True, num_samples=None):
    '''
    This runs several streaming PCA algorithms simultaneously on data that is provided in array X
    '''
    n, d = X.shape

    if num_samples is not None:
        num_acc = int(n / num_samples * num_acc)
        nblock = int(num_samples / B)
        endBsize = num_samples - nblock * B
    else:
        nblock = int(n / B)
        endBsize = n - nblock * B
    adaoja = stsb.AdaOja(d, k, b0=b0, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)
    hpca = stsb.HPCA(d, k, B=B, m=m, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

    for i in range(0, nblock*B, B):
        Xi = X[i:i+B]
        if endBsize == 0 and i == (nblock - 1) * B:
            adaoja.add_block(Xi, final_sample=True)
            hpca.add_block(Xi, final_sample=True)


        else:
            adaoja.add_block(Xi)
            hpca.add_block(Xi)


    if endBsize > 0:
        if num_samples is not None:
            Xi = X[nblock * B:num_samples]
        else:
            Xi = X[nblock * B:]
        adaoja.add_block(Xi, final_sample=True)
        hpca.add_block(Xi, final_sample=True)
    return adaoja, hpca


def run_sim_blocklist(Xlist, k, b0=1e-5, c_lin=1, c_sqrt=1, m=1, Sparse=True, Acc=True, xnorm2=None, num_acc=100, Time=True):
    '''
    This runs several streaming PCA methods simultaneously on a dataset provided as a list of blocks
    '''
    B, d = Xlist[0].shape
    adaoja = stsb.AdaOja(d, k, b0=b0, B=B, Sparse=Sparse, Acc=Acc, X=Xlist, xnorm2=xnorm2, num_acc=num_acc, Time=Time)
    hpca = stsb.HPCA(d, k, B=B, m=m, Sparse=Sparse, Acc=Acc, X=Xlist, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

    nblocks = len(Xlist)
    for i in range(nblocks-1):
        adaoja.add_block(Xlist[i])
        hpca.add_block(Xlist[i])

    adaoja.add_block(Xlist[-1], final_sample=True)
    hpca.add_block(Xlist[-1], final_sample=True)

    return adaoja, hpca

def run_adaoja_fullX(X, k, b0=1e-5, B=10, Sparse=False, Acc=True, xnorm2=None, num_acc=100, Time=True, num_samples=None):
    '''
    This runs AdaOja on data that is provided in array X
    '''
    n, d = X.shape
    if num_samples is not None:
        num_acc = int(n / num_samples * num_acc)
        nblock = int(num_samples / B)
        endBsize = num_samples - nblock * B
    else:
        nblock = int(n / B)
        endBsize = n - nblock * B

    adaoja = stsb.AdaOja(d, k, b0=b0, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

    for i in range(0, nblock*B, B):
        Xi = X[i:i+B]
        if endBsize == 0 and i == (nblock - 1) * B:
            adaoja.add_block(Xi, final_sample=True)
        else:
            adaoja.add_block(Xi)

    if endBsize > 0:
        if num_samples is not None:
            Xi = X[nblock * B:num_samples]
        else:
            Xi = X[nblock * B:]
        adaoja.add_block(Xi, final_sample=True)
    return adaoja
