import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from scipy import sparse as sp
from time import time
import scipy.sparse.linalg as spla
from math import sqrt

def list_exp_var(Xlist, Q, xnorm2):
    '''
    Obtains the explained variance for a dataset X given as a list of blocks.
    Inputs:
    ---------------------------
    Xlist: a list of Bxd arrays (the final element of the list may have different
        first dimension depending on n). This makes up a n x d dataset.
    Q: a dense d x k array. The vectors for which you are computing the
        explained variance.
    xnorm2: float > 0, the squared frobenius norm of X.

    Outputs:
    ----------------------------
    expvar: float > 0, the explained variance of Q for data X.
    '''
    XQnorm2 = 0
    for block in Xlist:
        XQnorm2 += la.norm(block.dot(Q), ord='fro')**2
    expvar = XQnorm2 / xnorm2
    return expvar

def list_xnorm2(Xlist, ord='fro', Sparse=True):
    '''
    Obtains the frobenius norm squared for a dataset X given as a list of blocks
    Inputs:
    ---------------------------
    Xlist: a list of Bxd arrays making up a n x d dataset X. The final element
        of the list may have different first dimension depending on n.
    ord: optional str or int, the order of the norm to compute. Default 'fro',
        the frobenius norm.
    Sparse: optional bool, default True. Indicates whether the elements of Xlist
        are sparse or dense.

    Outputs:
    ---------------------------
    xnorm2: float > 0, the ord norm squared of the data X contained in Xlist.
    '''
    xnorm2 = 0
    for Xi in Xlist:
        if Sparse:
            xnorm2 += spla.norm(Xi, ord=ord)**2
        else:
            xnorm2 += la.norm(Xi, ord=ord)**2
    return xnorm2

def list_mean(Xlist):
    # Obtain the mean for each column in a dataset expressed as a list of blocks.
    B, d = Xlist[0].shape
    means = np.zeros(d)
    sample_num = 0
    for Xi in Xlist:
        means += Xi.mean(axis=0)
        sample_num += Xi.shape[0]
    means /= sample_num
    return means

def scale_list(Xlist):
    # Subtract the mean value from a dataset split into a list of blocks.
    means = list_mean(Xlist)
    block_num = len(Xlist)
    for i in range(block_num):
        Xlist[i] -= means
    return Xlist

class StreamingPCA(object):
    def __init__(self, d, k, B=10, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False):
        '''
        Initialize the Streaming Oja's object.
            d: int > 0, the number of features in the dataset
            k: int > 0, the desired number of features
                (number of eigenvectors of the covariance matrix) to compute
            B: optional int > 0, default 1. The size of the blocks to manipulate
            Sparse: optional Bool, default False. Indicates whether the samples
                are added in as sparse or dense arrays.
            Acc: optional Bool, default False. Indicates whether the accuracy,
                here the explained variance, is computed at each block step
            X: Nonetype, nxd array, or list of Bval x d blocks Xi s.t. Xi make
                up the rows of X (note the last block in X may not be of length
                Bval, but all other blocks are assumed to have the same number
                of rows). X must be provided if Acc=True.
            xnorm2: The squared frobenius norm of X.
            num_acc: optional number of accuracy readings to take out of all
                possible block samples. Acc <= int(n/B)
            Time: optional Bool, default False. Indicates whether or not to time
                the implementation
        '''
        # Check to make sure the type requirements are fulfilled
        if not float(d).is_integer():
            raise TypeError('d must be an integer')
        if not float(k).is_integer():
            raise TypeError('k must be an integer')
        if not float(B).is_integer():
            raise TypeError('B must be an integer')
        if not float(num_acc).is_integer():
            raise TypeError('num_acc must be an integer')
        if not type(Acc) is bool:
            raise TypeError('Acc must be a boolean')
        if not type(Sparse) is bool:
            raise TypeError('Sparse must be a boolean')
        if not type(Time) is bool:
            raise TypeError('Time must be a boolean')

        # Make sure parameters fulfill value requirements
        if d <= 0:
            raise ValueError('d must be greater than 0')
        if k <= 0 or k > d:
            raise ValueError('must have 0 < k < d')
        if B <= 0:
            raise ValueError('B must be an integer greater than 0')
        if num_acc < 0:
            raise ValueError('num_acc must be nonnegative')
        # Set the initial parameters
        self.d, self.k, self.B, self.Acc, self.Sparse, self.Time = d, k, B, Acc, Sparse, Time

        # If the number of accuracy readings is zero, then no accuracy is taken and we set self.Acc = False
        if num_acc == 0:
            self.Acc = False

        # Set the sample number
        self.sample_num = 0
        # Set the block number
        self.block_num = 0
        # Initialize a list of stepsize values
        self.stepvals = []

        # Initialize Q1 (initial eigenvector guess)
        self.Q = la.qr(np.random.normal(size=(self.d, self.k)), mode='economic')[0]

        if self.Acc:
            self.acc_init(X, num_acc)
            self.xnorm2_init(xnorm2)

                # Initialize the components
        if self.Time:
            # initialize the list of update times
            self.time_vals = []
            self.total_time = 0

    def acc_init(self, X, num_acc):
        '''
        Initializes the class structures needed to compute the accuracy for
            num_acc number of steps
        '''
        # Initialize the list of accuracies
        self.accQ = []
        # Make sure X has been initialized
        if X is None:
            raise ValueError('To compute the explained variance at each step, the array of samples must be provided')

        # Check to see if X is an ndarray or a list:
        if type(X) is list:
            self.islist = True
            self.X = X
            Bval, dval = self.X[0].shape
            self.n = (len(self.X) - 1) * Bval + self.X[-1].shape[0]
        else:
            self.islist = False
            self.X = X
            self.n, dval = X.shape

        # make sure the number of features of X matches self.d
        if dval != self.d:
            raise ValueError('Column number of X, d=' + str(dval) + ' does not match self.d=' + str(self.d))

        # Check to make sure the number of accuracy readings isn't greater than the total number of blocks.
        if num_acc > int(self.n / self.B):
            print("Number of accuracy readings is greater than the number of samples. Setting num_acc to int(n/B)")
            self.num_acc = int(self.n / self.B)
        else:
            self.num_acc = num_acc

        # Find the blocksize for each accuracy measurement by taking the floor of n / num_acc
        self.accBsize = int(self.n / self.num_acc)
        self.acc_indices = [0]

        # Current accuracy number (indicates for which sample numbers the accuracy should be calculated.)
        # The accuracy will be calculated for the block containing that accuracy number. This is why we ensure the
        # number of accuracy computations is not greater than the number of blocks.
        self.curr_acc_num = self.accBsize

    def xnorm2_init(self, xnorm2):
        # Make sure xnorm2 is initalized and take the first accuracy reading
        self.xnorm2 = xnorm2
        if self.islist:
            # If the squared frobenius of X norm is not provided, calculate it yourself
            if xnorm2 is None:
                if self.Sparse:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=True)
                else:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=False)
            # Take an initial reading of the explained variance
            self.accQ.append(list_exp_var(self.X, self.Q, self.xnorm2))
        else:
            # If the squared frobenius of X norm is not provided, calculate it yourself
            if xnorm2 is None:
                if self.Sparse:
                    self.xnorm2 = spla.norm(self.X, ord='fro')**2
                else:
                    self.xnorm2 = la.norm(self.X, ord='fro')**2
            # Take an initial reading of the explained variance
            self.accQ.append(la.norm(self.X.dot(self.Q), ord='fro')**2 / self.xnorm2)

    def add_block(self, Xi, final_sample=False):
        '''
        Adds B x d block of samples Xi to the model.
        Note in this method we want to make sure B is consistent at each step
        but it would be easy to modify the code to allow for varied B values
        over time.

        Inputs:
            Xi: B x d block of samples. If self.Sparse = True, this is a sparse
                array of samples, otherwise it is dense.
            final sample: optional boolean, default false. Indicates whether the
                sample should be treated as the final sample.
        '''
        Bval, dval = Xi.shape

        if not final_sample:
            # Check to make sure the blocksize and dimensions are consistent with
            # the model. To allow for varying B, we would remove the Bval check.
            if Bval != self.B or dval != self.d:
                raise ValueError("(" + str(Bval) + ", " + str(dval) +") does not match (B,d) = (" +str(self.B) + ", " + str(self.d) + ")")

            # Increase the sample number and block number.
            self.sample_num += self.B
            self.block_num += 1
        else:
            if dval != self.d:
                raise ValueError("Sample dimension " + str(dval) + " != " + str(d))
            self.sample_num += Bval
            self.block_num += (Bval / self.B)

        # Set the current sample block if it passed the checks
        self.Xi = Xi

        # Take a first time reading if Time
        if self.Time:
            time0 = time()

        # Perform the update step for sparse or dense data, respectively
        if self.Sparse:
            self.sparse_update()
        else:
            self.dense_update()

        # Note the timing here is only for the update step
        if self.Time:
            time1 = time()
            self.total_time += (time1 - time0)
            self.time_vals.append(self.total_time)

        # Calculate the accuracy if the next "curr_acc_num" falls in this block.
        if self.Acc and (self.sample_num - self.B <= self.curr_acc_num < self.sample_num or final_sample):
            # Calculate the accuracy
            self.get_Acc(final_sample)

    def get_Acc(self, final_sample):
        '''
        Calculates the accuracy for the current set of vectors self.Q
        '''
        # Calculate the accuracy
        if self.islist:
            self.accQ.append(list_exp_var(self.X, self.Q, self.xnorm2))
        else:
            self.accQ.append(la.norm(self.X.dot(self.Q), ord='fro')**2 / self.xnorm2)

        # Update the current accuracy number
        self.curr_acc_num += self.accBsize

        # Append the block number associated with the accuracy reading
        if final_sample:
            self.acc_indices.append(self.n)
        else:
            self.acc_indices.append(self.sample_num)

    def plot(self, dataname):
        '''
        Plots the explained variance and/or the time depending on whether they
        were collected.
        Inputs:
            dataname: str, name for the data
        '''
        if not type(dataname) is str:
            raise TypeError('dataname must be a str')
        if self.Acc:
            # plot explained variance
            plt.plot(self.acc_indices, self.accQ)
            plt.title("Explained variance for " + dataname)
            plt.xlabel("Approximate number of samples")
            plt.ylabel("Explained variance")
            plt.show()

        if self.Time:
            plt.plot(np.arange(0, self.n, self.B), self.time_vals)
            plt.xlabel('Number of samples')
            plt.ylabel('Total time in seconds')
            plt.title("Algorithm update time for " + dataname)
            plt.show()

class Oja(StreamingPCA):
    '''
    Class for Oja's streaming PCA method with learning rate c/sqrt(t) or c/t.
    '''
    def __init__(self, d, k, c=1., B=10, Sqrt=True, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False, single_acc_B_index=10):
        '''
        Initialize the Oja's method object.
        c: optional float, the constant that determines the scale of the
            learning rate. Default 1.
        Sqrt: optional bool, default True. This defines the learning rate choice
            for Oja's method. If Sqrt = True, then the learning rate will be
            c/sqrt(t), otherwise it will be set to c/t.

        '''
        StreamingPCA.__init__(self, d, k, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

        if not float(single_acc_B_index).is_integer():
            raise TypeError('single_acc_B_index must be an integer')
        if not type(Sqrt) is bool:
            raise TypeError('Sqrt must be a boolean')
        if single_acc_B_index < 0:
            raise ValueError('single_acc_B_index must be nonnegative')

        self.c, self.Sqrt, self.single_acc_B_index = c, Sqrt, single_acc_B_index


    def add_block(self, Xi, final_sample=False):
        StreamingPCA.add_block(self, Xi, final_sample=final_sample)
        if self.Acc:
            if self.block_num == self.single_acc_B_index and self.num_acc==1:
                StreamingPCA.get_Acc(self, final_sample)

    def dense_update(self):
        if self.Sqrt:
            eta = self.c / sqrt(self.sample_num)
        else:
            eta = self.c / self.sample_num
        self.Q += eta * self.Xi.T @ (self.Xi @ self.Q) / self.B
        self.Q = la.qr(self.Q, mode='economic')[0]

    def sparse_update(self):
        if self.Sqrt:
            eta = self.c / sqrt(self.sample_num)
        else:
            eta = self.c / self.sample_num

        self.Q += eta * self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
        self.Q = la.qr(self.Q, mode='economic')[0]

class AdaOja(StreamingPCA):
    '''
    Implements the AdaOja algorithm with vector of learning rates.
    '''
    def __init__(self, d, k, b0=1e-5, B=10, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False, update_norm=2, single_acc_B_index=10):
        '''
        b0: optional float, default 1e-5. The initial "guess" for the learning
            rate parameter in adagrad.
        update_norm: optional parameter. Indicates the order of the norm used to
            compute the learning rate. Default 2.
        '''
        StreamingPCA.__init__(self, d, k, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)
        if not float(single_acc_B_index).is_integer():
            raise TypeError('single_acc_B_index must be an integer')
        if b0 < 0:
            raise ValueError('b0 must be nonnegative')
        if single_acc_B_index < 0:
            raise ValueError('single_acc_B_index must be nonnegative')

        self.unorm, self.single_acc_B_index = update_norm, single_acc_B_index
        self.b0 = np.ones(self.k) * b0
        self.stepvals = [1 / self.b0]

    def add_block(self, Xi, final_sample=False):
        StreamingPCA.add_block(self, Xi, final_sample=final_sample)
        if self.Acc:
            if self.block_num == self.single_acc_B_index and self.num_acc==1:
                StreamingPCA.get_Acc(self, final_sample)

    def dense_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T @ (self.Xi @ self.Q) / self.B
        self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        self.stepvals.append(1/self.b0)
        self.Q += G / self.b0
        self.Q = la.qr(self.Q, mode='economic')[0]

    def sparse_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
        self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        self.stepvals.append(1/self.b0)
        self.Q += G / self.b0
        self.Q = la.qr(self.Q, mode='economic')[0]


class HPCA(StreamingPCA):
    '''
    Implements the history PCA method from "Histoy PCA: a New Algorithm for
        Streaming PCA" by Yang, Hsieh and Wang.
    '''
    def __init__(self, d, k, B=10, m=1, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False):
        '''
        m: optional int > 0, default 1. The number of convergence iterations
            per block.
        '''
        StreamingPCA.__init__(self, d, k, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)
        self.m = m
        self.S1 = np.random.normal(size=(d,k))
        self.lam = np.zeros((k,1))

    def dense_update(self):
        # Make a local variable for the current value of Q1
        Q0 = np.copy(self.Q)
        for j in range(self.m):
            self.S1 = (self.block_num - 1) / self.block_num * Q0 @ ((self.lam * Q0.T) @ self.Q) + self.Xi.T @ (self.Xi @ self.Q) / self.sample_num
            self.Q = la.qr(self.S1, mode='economic')[0]
        self.lam[:,0] = la.norm(self.S1, axis=0)

    def sparse_update(self):
        # Make a local variable for the current value of Q1
        Q0 = np.copy(self.Q)

        for j in range(self.m):
            self.S1 = (self.block_num - 1) / self.block_num * Q0 @ ((self.lam * Q0.T) @ self.Q) + self.Xi.T.dot(self.Xi.dot(self.Q)) / self.sample_num
            self.Q = la.qr(self.S1, mode='economic')[0]
        self.lam[:,0] = la.norm(self.S1, axis=0)

class SPM(StreamingPCA):
    '''
    Implements the block power method found in "Memory Limited, Streaming PCA"
        by Mitliagkas, Caramanis, and Jain
    '''
    def __init__(self, d, k, p, B=10, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False):
        assert p >= k, "p must be >= k"
        # Note we initialize our method with p rather than k, becaues this
        # method can apparently obtain better convergence if a few extra vectors
        # are computed, then truncated to k.
        self.true_k, self.p = k, p
        StreamingPCA.__init__(self, d, p, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

    def xnorm2_init(self, xnorm2):
        # Make sure xnorm2 is initalized and take the first accuracy reading
        self.xnorm2 = xnorm2
        if self.islist:
            # If the squared frobenius of X norm is not provided, calculate it yourself
            if xnorm2 is None:
                if self.Sparse:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=True)
                else:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=False)
            # Take an initial reading of the explained variance
            self.accQ.append(list_exp_var(self.X, self.Q[:,:self.true_k], self.xnorm2))
        else:
            # If the squared frobenius of X norm is not provided, calculate it yourself
            if xnorm2 is None:
                if self.Sparse:
                    self.xnorm2 = spla.norm(self.X, ord='fro')**2
                else:
                    self.xnorm2 = la.norm(self.X, ord='fro')**2
            # Take an initial reading of the explained variance
            self.accQ.append(la.norm(self.X.dot(self.Q[:,:self.true_k]), ord='fro')**2 / self.xnorm2)

    def get_Acc(self, final_sample):
        '''
        Calculates the accuracy for the current set of vectors
        self.Q[:,:self.true_k]. Note we must necessarily redefine this function
        since the algorithm calculates extra vectors.
        '''
        # Calculate the accuracy
        if self.islist:
            self.accQ.append(list_exp_var(self.X, self.Q[:,:self.true_k], self.xnorm2))
        else:
            self.accQ.append(la.norm(self.X.dot(self.Q[:,:self.true_k]), ord='fro')**2 / self.xnorm2)

        # Update the current accuracy number
        self.curr_acc_num += self.accBsize

        # Append the block number associated with the accuracy reading
        if final_sample:
            self.acc_indices.append(self.n)
        else:
            self.acc_indices.append(self.sample_num)



    def dense_update(self):
        S = self.Xi.T @ (self.Xi @ self.Q)
        self.Q = la.qr(S, mode='economic')[0]

    def sparse_update(self):
        S = self.Xi.T.dot(self.Xi.dot(self.Q))
        self.Q = la.qr(S, mode='economic')[0]

class PM_momentum(StreamingPCA):
    '''
    Implements the Mini-batch Power Method with Momentum found in "Accelerated
        Stochastic Power Iteration" by De Sa, He, Mitliagkas, Re, and Xu.
    '''
    def __init__(self, d, k, beta, B=10, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False):
        StreamingPCA.__init__(d, k, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)
    def dense_update(self):
        pass
    def sparse_update(self):
        pass
