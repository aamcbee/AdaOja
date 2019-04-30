import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from scipy import sparse as sp
from time import time
import scipy.sparse.linalg as spla
from math import sqrt
import streaming_subclass as stsb


def plot_hpca_ada(adaoja, hpca, data_name, fig_name=None, true_evar=None):
    '''
    Plots and saves the explained variance for AdaOja vs HPCA for a given dataset.
    Input:
        adaoja: an AdaOja class object (see streaming_subclass.py for details)
        hpca: an HPCA class object (see streaming_subclass.py for details)
        data_name: str, name for the data being applied
        fig_name: None or optional string, the name of the figure to be saved.
            if fig_name = None, default fig_name is 'hpcavada_' + data_name +
            '_k' + str(k) + '.png'.
        true_evar: None or optional positive float between 0 and 1. The
            explained variance for the top k true eigenvectors of the covariance
            matrix (typically the sample covariance matrix). This allows us to
            compare our methods to the "best case", offline result.
    '''
    k = adaoja.k
    plt.plot(adaoja.acc_indices, adaoja.accQ, '--', color='green', label='AdaOja')
    plt.plot(hpca.acc_indices, hpca.accQ, '-.', color='black', label='HPCA')

    if true_evar is not None:
        assert true_evar >= 0 and true_evar <=1, "The true explained variance should be a float > 0"
        plt.plot(adaoja.acc_indices, np.ones_like(adaoja.acc_indices) * true_evar, color='purple', label='Offline SVD')


    plt.legend(loc='best')
    plt.title('HPCA vs AdaOja\n' + data_name + ', k=' + str(k))
    plt.xlabel('Number of samples')
    plt.ylabel('Explained Variance')
    if fig_name is None:
        plt.savefig('hpcavada_' + data_name + '_k' + str(k) + '.png')
    else:
        plt.savefig(fig_name)
    plt.show()

class compare_lr(object):
    def __init__(self, base=2., lower=-10, upper=10, test_index=0):
        '''
        Initializes an object to compare Oja to AdaOja for different learning
        rates
        Inputs:
            base: optional float > 0, the base for the chosen c values.
                Default 2.
            lower: optional float, determines the lower bound base^lower <= c
            upper: optional float, determines the upper bound base^upper >= c.
                Note c will range between [base^lower, ... base^{upper}]
            test_index: optional int > 0, default is 0. If test_index > 0, then
                the accuracy at the test_index will be reported for each c value.
                If test_index=0, then the accuracy at the final step will be
                reported for each c value.

        '''
        self.base, self.lower, self.upper, self.test_index = base, lower, upper, test_index
        self.n = None

        # Initialize c parameters
        self.cvals = base**np.arange(lower, upper)

    def run_cval_bag(self, filename, k, B=10, b0=1e-5, m=1, Sparse=True, X=None, xnorm2=None):
        '''
        This runs several streaming PCA algorithms simultaneously on bag of
        words data for a range of constant values c.

        Inputs:
        ----------------------------------------------
        filename: The name of the file containing the bag-of-words data
        k: int, the number of top eigenvectors to compute using the streaming
            PCA algorithms
        B: optional int, the batch size for the streaming methods
        b0: optional float > 0, default 1e-5. The initial "guess" for the
            learning rate parameter for adagrad
        Sparse: optional Bool, default False. Indicates whether the samples are
            added in as sparse or dense arrays.
        X: Nonetype, nxd array, or list of Bval x d blocks Xi s.t. Xi make up
            the rows of X (note the last block in X may not be of length Bval,
            but all other blocks are assumed to have the same number of rows).
            X must be provided if Acc=True.
        xnorm2: optional float, the squared frobenius norm of the full dataset

        Produces:
        ---------------------------------------------------------
        self.lin_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/t for c in self.cvals
        self.sqrt_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/sqrt(t) for c in self.cvals
        self.adaoja_acc: the single accuracy value for adaoja on the given
            dataset (since this doesn't tune hyperparameters)
        self.adaoja: StreamingPCA AdaOja object
        '''
        self.B = B
        with open(filename, 'r') as f:
            self.n = int(f.readline())
            self.d = int(f.readline())
            nnz = int(f.readline())

            # Initialize a list of oja objects
            lin_list = [stsb.Oja(self.d, k, c=c, B=self.B, Sqrt=False, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
            sqrt_list = [stsb.Oja(self.d, k, c=c, B=self.B, Sqrt=True, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
            self.adaoja = stsb.AdaOja(self.d, k, b0=b0, B=self.B, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index)

            blocknum = 1
            row = []
            col = []
            data = []
            for i in range(nnz):

                entry = list(map(int, f.readline().split()))
                # if the row num (with zero based indexing)
                # is in the current block
                if entry[0] - 1 < blocknum * self.B:
                    # note bag of words uses 1 based indexing
                    row.append((entry[0]-1) % self.B)
                    col.append(entry[1]-1)
                    data.append(entry[2])
                else:
                    # Add the current block to the model
                    if Sparse:
                        Xi = sp.csr_matrix((data, (row, col)), shape=(self.B, self.d))
                    else:
                        Xi = np.zeros((self.B, self.d))
                        Xi[row, col] = data

                    for oja_lin in lin_list:
                        oja_lin.add_block(Xi)
                    for oja_sqrt in sqrt_list:
                        oja_sqrt.add_block(Xi)

                    self.adaoja.add_block(Xi)
                    # Increase the block number
                    blocknum += 1
                    # Start the new block in the row, col, and data entries.
                    row = [(entry[0] - 1) % self.B]
                    col = [entry[1] - 1]
                    data = [entry[2]]
            # Insert final block
            if Sparse:
                Xi = sp.csr_matrix((data, (row, col)), shape=(max(row) + 1, self.d))
            else:
                Xi = np.zeros((max(row) + 1, self.d))
                Xi[row,col] = data
            for oja_lin in lin_list:
                oja_lin.add_block(Xi, final_sample=True)
            for oja_sqrt in sqrt_list:
                oja_sqrt.add_block(Xi, final_sample=True)
            self.adaoja.add_block(Xi, final_sample=True)

            # If the desired index to return is not the first accuracy metric return it
            # Otherwise, return the final accuracy achieved
            if self.test_index > 0:
                self.lin_acc = [oja_lin.accQ[-2] for oja_lin in lin_list]
                self.sqrt_acc = [oja_sqrt.accQ[-2] for oja_sqrt in sqrt_list]
                self.adaoja_acc = self.adaoja.accQ[-2]
            else:
                self.lin_acc = [oja_lin.accQ[-1] for oja_lin in lin_list]
                self.sqrt_acc = [oja_sqrt.accQ[-1] for oja_sqrt in sqrt_list]
                self.adaoja_acc = self.adaoja.accQ[-1]

    def run_cval_fullX(self, X, k, B=10, b0=1e-5, m=1, Sparse=False, xnorm2=None):
        '''
        This runs several streaming PCA algorithms simultaneously on data that
        is provided in array X
        Inputs:
        ----------------------------------------------
        X: an n x d array of data, can be sparse or dense (see Sparse boolean
            parameter).
        k: int, the number of top eigenvectors to compute using the streaming
            PCA algorithms
        B: optional int, the batch size for the streaming methods, default 10.
        b0: optional float > 0, default 1e-5. The initial "guess" for the
            learning rate parameter for AdaOja.
        m: optional int > 0, default 1. The number of convergence iterations per
            block.
        Sparse: optional Bool, default False. Indicates whether the samples are
            added in as sparse or dense arrays.
        xnorm2: optional float, the squared frobenius norm of X. Used in
            accuracy calculation.


        Produces:
        ---------------------------------------------------------
        self.lin_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/t for c in self.cvals
        self.sqrt_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/sqrt(t) for c in self.cvals
        self.adaoja_acc: the single accuracy value for adaoja on the given
            dataset (since this doesn't tune hyperparameters)
        self.adaoja: StreamingPCA AdaOja object
        '''
        self.B = B
        self.n, self.d = X.shape

        # Initialize a list of oja objects
        lin_list = [stsb.Oja(self.d, k, c=c, B=self.B, Sqrt=False, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
        sqrt_list = [stsb.Oja(self.d, k, c=c, B=B, Sqrt=True, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
        self.adaoja = stsb.AdaOja(self.d, k, b0=b0, B=B, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=X, num_acc=1, Time=False, single_acc_B_index=self.test_index)

        nblock = int(self.n / self.B)
        endBsize = self.n - nblock * self.B
        for i in range(0, nblock*self.B, self.B):
            Xi = X[i:i+self.B]
            if endBsize == 0 and i == (nblock - 1) * self.B:
                for oja_lin in lin_list:
                    oja_lin.add_block(Xi, final_sample=True)
                for oja_sqrt in sqrt_list:
                    oja_sqrt.add_block(Xi, final_sample=True)
                self.adaoja.add_block(Xi, final_sample=True)

            else:
                for oja_lin in lin_list:
                    oja_lin.add_block(Xi)
                for oja_sqrt in sqrt_list:
                    oja_sqrt.add_block(Xi)
                self.adaoja.add_block(Xi)

        if endBsize > 0:
            Xi = X[nblock * self.B:]
            for oja_lin in lin_list:
                oja_lin.add_block(Xi, final_sample=True)
            for oja_sqrt in sqrt_list:
                oja_sqrt.add_block(Xi, final_sample=True)
            self.adaoja.add_block(Xi, final_sample=True)

        # If the desired index to return is not the first accuracy metric return it
        # Otherwise, return the final accuracy achieved
        if self.test_index > 0:
            self.lin_acc = [oja_lin.accQ[-2] for oja_lin in lin_list]
            self.sqrt_acc = [oja_sqrt.accQ[-2] for oja_sqrt in sqrt_list]
            self.adaoja_acc = self.adaoja.accQ[-2]
        else:
            self.lin_acc = [oja_lin.accQ[-1] for oja_lin in lin_list]
            self.sqrt_acc = [oja_sqrt.accQ[-1] for oja_sqrt in sqrt_list]
            self.adaoja_acc = self.adaoja.accQ[-1]

    def run_cval_blocklist(self, Xlist, k, b0=1e-5, m=1, Sparse=True, xnorm2=None):
        '''
        This runs several streaming PCA methods simultaneously on a dataset
        provided as a list of blocks

        Inputs:
        ----------------------------------------------
        Xlist: A list of B x d datablocks that make up the dataset. Note the
            final block may not be B x d if n % d > 0.
        k: int, the number of top eigenvectors to compute using the streaming
            PCA algorithms
        b0: optional float > 0, default 1e-5. The initial "guess" for the
            learning rate parameter for adagrad
        m: optional int > 0, default 1. The number of convergence iterations
                per block.
        Sparse: optional Bool, default False. Indicates whether the samples are
            added in as sparse or dense arrays.
        xnorm2: optional float, the squared frobenius norm of X.

        Produces:
        ---------------------------------------------------------
        self.lin_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/t for c in self.cvals
        self.sqrt_acc: the list of final accuracies (or test_index accuracies)
            for Oja's method with stepsize c/sqrt(t) for c in self.cvals
        self.adaoja_acc: the single accuracy value for adaoja on the given
            dataset (since this doesn't tune hyperparameters)
        self.adaoja: StreamingPCA AdaOja object
        '''

        self.B, self.d = Xlist[0].shape

        # Initialize a list of oja objects
        lin_list = [stsb.Oja(self.d, k, c=c, B=self.B, Sqrt=False, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=Xlist, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
        sqrt_list = [stsb.Oja(self.d, k, c=c, B=self.B, Sqrt=True, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=Xlist, num_acc=1, Time=False, single_acc_B_index=self.test_index) for c in self.cvals]
        self.adaoja = stsb.AdaOja(self.d, k, b0=b0, B=self.B, Sparse=Sparse, Acc=True, xnorm2=xnorm2, X=Xlist, num_acc=1, Time=False, single_acc_B_index=self.test_index)

        nblocks = len(Xlist)
        for i in range(nblocks-1):
            for oja_lin in lin_list:
                oja_lin.add_block(Xlist[i])
            for oja_sqrt in sqrt_list:
                oja_sqrt.add_block(Xlist[i])
            self.adaoja.add_block(Xlist[i])
        for oja_lin in lin_list:
            oja_lin.add_block(Xlist[-1], final_sample=True)
        for oja_sqrt in sqrt_list:
            oja_sqrt.add_block(Xlist[-1], final_sample=True)
        self.adaoja.add_block(Xlist[-1], final_sample=True)

        # If the desired index to return is not the first accuracy metric return it
        # Otherwise, return the final accuracy achieved
        if self.test_index > 0:
            self.lin_acc = [oja_lin.accQ[-2] for oja_lin in lin_list]
            self.sqrt_acc = [oja_sqrt.accQ[-2] for oja_sqrt in sqrt_list]
            self.adaoja_acc = self.adaoja.accQ[-2]
        else:
            self.lin_acc = [oja_lin.accQ[-1] for oja_lin in lin_list]
            self.sqrt_acc = [oja_sqrt.accQ[-1] for oja_sqrt in sqrt_list]
            self.adaoja_acc = self.adaoja.accQ[-1]

    def plot_cval_acc(self, dataname, figname, true_evar = None):
        '''
        Plot self.lin_acc, self.sqrt_acc, and self.adaoja_acc against each other.

        Inputs:
        --------------------------------
        dataname: str, the name of the dataset
        figname: str, the name of the figure to save. Typically ends in .png
        true_evar: None or optional positive float between 0 and 1.
            The explained variance for the top k true eigenvectors of the
            covariance matrix (typically the sample covariance matrix). This
            allows us to compare our methods to the best case, offline result.
        '''
        assert self.lin_acc is not None, "Objects to plot have not yet been initialized"

        if self.n is None:
            self.n = self.adaoja.sample_num
        if self.test_index == 0:
            num_samples = self.n
        else:
            num_samples = self.B * self.test_index
        c_title = 'Explained variance after ' + str(num_samples) + ' samples, varying c\n' + dataname
        plt.scatter(np.log(self.cvals), self.lin_acc, label=r'Oja, $\eta_i = c/t$')
        plt.scatter(np.log(self.cvals), self.sqrt_acc, marker='+', color='k', label=r'Oja, $\eta_i = c/\sqrt{t}$')
        plt.plot(np.log(self.cvals), np.ones_like(self.cvals) * self.adaoja_acc, '--', color='green', label='AdaOja')

        # If the true explained variance is given, plot it
        if true_evar is not None:
            assert true_evar >= 0 and true_evar <=1, "The true explained variance should be a float > 0"
            plt.plot(np.log(self.cvals), np.ones_like(self.cvals) * true_evar, color='purple', label='Offline SVD')

        plt.xlabel('log(c)')
        plt.ylabel('Explained Variance')
        plt.title('Explained variance after ' + str(num_samples) + ' samples, varying c\n' + dataname)
        plt.legend(loc='best')
        plt.savefig(figname)
        plt.show()

    def plot_bvals(self, dataname, figname, loglog=True):
        '''
        Plots the learning rates generated by AdaOja against the best case c/t
        and c/sqrt(t) learning rates.

        Inputs:
        --------------------------------
        dataname: str, the name of the dataset
        figname: str, the name of the figure to save. Typically ends in .png
        loglog: optional bool, default True. Indicates whether to plot loglog
        scale
        '''
        title = ('Learning Rates for Oja vs AdaOja\n' + dataname)

        self.clin, self.csq = self.cvals[np.argmax(self.lin_acc)], self.cvals[np.argmax(self.sqrt_acc)]
        adaoja_bvals = np.array(self.adaoja.stepvals)
        if self.adaoja.sample_num % self.B == 0:
            samplenum = np.arange(self.B, self.adaoja.sample_num +1, self.B)
        else:
            samplenum = np.hstack((np.arange(self.B, self.adaoja.sample_num, self.B), self.adaoja.sample_num))
        if loglog:
            plt.loglog(samplenum, self.clin / samplenum, '-.', label=str(self.clin) + r'$/ t$')
            plt.loglog(samplenum, self.csq / np.sqrt(samplenum), ':', color='black', label=str(self.csq) + r'$/\sqrt{t}$')

            if self.adaoja.k > 1:
                plt.loglog(samplenum, adaoja_bvals[1:,0], '-', color='green', label='AdaOja ' + r'$ 1/b_t[0]$')
                plt.loglog(samplenum, adaoja_bvals.mean(axis=1)[1:], '--', color='Green', label='AdaOja Avg ' + r'1/$b_t[i]$')
            else:
                plt.loglog(samplenum, adaoja_bvals[1:,0], '-', color='green', label='AdaOja ' + r'$ 1/b_t$')
        else:
            plt.plot(samplenum, self.clin / samplenum, '-.', label=str(self.clin) + r'$/ t$')
            plt.plot(samplenum, self.csq / np.sqrt(samplenum), ':', color='black', label=str(self.csq) + r'$/\sqrt{t}$')
            if self.adaoja.k > 1:
                plt.plot(samplenum, adaoja_bvals[1:,0], '-', color='Green', label='AdaOja ' + r'$ 1/b_t[0]$')
                plt.plot(samplenum, adaoja_bvals.mean(axis=1)[1:], '--', color='Green', label='AdaOja Avg ' + r'1/$b_t[i]$')
            else:
                plt.plot(samplenum, adaoja_bvals[1:,0], '-', color='Green', label='AdaOja ' + r'$ 1/b_t$')
        plt.legend(loc='best')
        plt.xlabel('Number of Samples')
        plt.ylabel('Learning Rate')
        plt.title(title)
        plt.savefig(figname)
        plt.show()
