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
                of rows). X must be provided if Acc=True so the accuracy can be
                computed.
            xnorm2: optional float > 0, the squared frobenius norm of X.
            num_acc: optional int, the number of accuracy readings to take out of all
                possible block samples. We require num_acc <= int(n/B). Default 100.
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
        Calculates the accuracy for the current set of vectors self.Q.
        Accuracy here is defined to be the explained variance.
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
    def __init__(self, *args, c=1., Sqrt=True, single_acc_B_index=10, **kwargs):
        '''
        Initialize the Oja's method object.
        c: optional float, the constant that determines the scale of the
            learning rate. Default 1.
        Sqrt: optional bool, default True. This defines the learning rate choice
            for Oja's method. If Sqrt = True, then the learning rate will be
            c/sqrt(t), otherwise it will be set to c/t.

        '''
        super().__init__(*args, **kwargs)
        #StreamingPCA.__init__(self, d, k, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

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
    Implements the AdaOja algorithm with a vector of learning rates.
    '''
    def __init__(self, *args, b0=1e-5, unorm=2, single_acc_B_index=10, b0_dim=1, **kwargs):
        '''
        b0: optional float, default 1e-5. The initial "guess" for the learning
            rate parameter in adagrad.
        unorm: optional parameter. Indicates the order of the norm used to
            compute the learning rate. Default 2.
        '''
        super().__init__(*args, **kwargs)
        if not float(single_acc_B_index).is_integer():
            raise TypeError('single_acc_B_index must be an integer')
        if b0 < 0:
            raise ValueError('b0 must be nonnegative')
        if single_acc_B_index < 0:
            raise ValueError('single_acc_B_index must be nonnegative')


        self.unorm, self.single_acc_B_index, self.b0_dim = unorm, single_acc_B_index, b0_dim
        if self.b0_dim==0:
            self.b0 = b0
        elif self.b0_dim ==1:
            self.b0 = np.ones(self.k) * b0
        elif self.b0_dim == 2:
            self.b0 = np.ones((self.d, self.k)) * b0
        else:
            raise ValueError("b0_dim options are 0: constant, 1: vector, or 2: matrix")

        self.stepvals = [1 / self.b0]
        self.cvals = [np.sqrt(self.sample_num) / self.b0]

    def add_block(self, Xi, final_sample=False):
        StreamingPCA.add_block(self, Xi, final_sample=final_sample)
        if self.Acc:
            if self.block_num == self.single_acc_B_index and self.num_acc==1:
                StreamingPCA.get_Acc(self, final_sample)

    def dense_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T @ (self.Xi @ self.Q) / self.B
        if self.b0_dim == 0:
            self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm)**2)
        if self.b0_dim == 1:
            self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        if self.b0_dim == 2:
            self.b0 = np.sqrt(self.b0**2 + G**2)
        self.stepvals.append(1/self.b0)
        self.cvals.append(np.sqrt(self.sample_num)/self.b0)
        self.Q += G / self.b0
        self.Q = la.qr(self.Q, mode='economic')[0]

    def sparse_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
        if self.b0_dim == 0:
            self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm)**2)
        if self.b0_dim == 1:
            self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        if self.b0_dim == 2:
            self.b0 = np.sqrt(self.b0**2 + G**2)
        self.stepvals.append(1/self.b0)
        self.cvals.append(np.sqrt(self.sample_num)/self.b0)
        self.Q += G / self.b0
        self.Q = la.qr(self.Q, mode='economic')[0]

class WindOja(StreamingPCA):
    '''
    Implements the AdaOja algorithm with a vector of learning rates.
    '''
    def __init__(self, *args, b0=1e-5, unorm=2, single_acc_B_index=10, b0_dim=1, tol=1e-3, **kwargs):
        '''
        b0: optional float, default 1e-5. The initial "guess" for the learning
            rate parameter in adagrad.
        unorm: optional parameter. Indicates the order of the norm used to
            compute the learning rate. Default 2.
        '''
        super().__init__(*args, **kwargs)
        if not float(single_acc_B_index).is_integer():
            raise TypeError('single_acc_B_index must be an integer')
        if b0 < 0:
            raise ValueError('b0 must be nonnegative')
        if single_acc_B_index < 0:
            raise ValueError('single_acc_B_index must be nonnegative')


        self.unorm, self.single_acc_B_index, self.b0_dim = unorm, single_acc_B_index, b0_dim
        if self.b0_dim==0:
            self.b0 = b0
        elif self.b0_dim ==1:
            self.b0 = np.ones(self.k) * b0
        elif self.b0_dim == 2:
            self.b0 = np.ones((self.d, self.k)) * b0
        else:
            raise ValueError("b0_dim options are 0: constant, 1: vector, or 2: matrix")

        self.stepvals = [1 / self.b0]
        self.cvals = [np.sqrt(self.sample_num) / self.b0]
        # Boolean to determine whether the window has been hit
        self.window = False

    def add_block(self, Xi, final_sample=False):
        StreamingPCA.add_block(self, Xi, final_sample=final_sample)
        if self.window and final_sample:
            self.Q = la.qr(self.Q, mode='economic')[0]
        if self.Acc:
            if self.block_num == self.single_acc_B_index and self.num_acc==1:
                StreamingPCA.get_Acc(self, final_sample)

    def dense_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)
        if self.window:
            pass
        else:
            G = self.Xi.T @ (self.Xi @ self.Q) / self.B
            if self.b0_dim == 0:
                self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm)**2)
            if self.b0_dim == 1:
                self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
            if self.b0_dim == 2:
                self.b0 = np.sqrt(self.b0**2 + G**2)
            self.stepvals.append(1/self.b0)
            self.cvals.append(np.sqrt(self.sample_num)/self.b0)
            self.eval_cval()
            self.Q += G / self.b0
            self.Q = la.qr(self.Q, mode='economic')[0]


    def sparse_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)
        if self.window:
            pass
        else:
            G = self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
            if self.b0_dim == 0:
                self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm)**2)
            if self.b0_dim == 1:
                self.b0 = np.sqrt(self.b0**2 + np.linalg.norm(G, ord=self.unorm, axis=0)**2)
            if self.b0_dim == 2:
                self.b0 = np.sqrt(self.b0**2 + G**2)
            self.stepvals.append(1/self.b0)
            self.cvals.append(np.sqrt(self.sample_num)/self.b0)
            self.eval_cval()
            self.Q += G / self.b0
            self.Q = la.qr(self.Q, mode='economic')[0]

    def eval_cval(self, size=3):
        if np.mean(np.array(self.cvals[-size:-1]) - np.array(self.cvals[-size-1:-1]))) < self.tol:
            self.window=True
            self.c = np.mean(self.cvals[-size:-1])





################################################################################
class HPCA(StreamingPCA):
    '''
    Implements the history PCA method from "Histoy PCA: a New Algorithm for
        Streaming PCA" by Yang, Hsieh and Wang.
    '''
    def __init__(self, *args, m=1, **kwargs):
        '''
        m: optional int > 0, default 1. The number of convergence iterations
            per block.
        '''
        super().__init__(*args, **kwargs)
        self.m = m
        self.S1 = np.random.normal(size=(self.d,self.k))
        self.lam = np.zeros((self.k,1))

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
    def __init__(self, d, k, p=None, B=10, Sparse=False, Acc=False, X=None, xnorm2=None, num_acc=100, Time=False):
        if p is None:
            p = k

        assert p >= k, "p must be >= k"
        # Note we initialize our method with p rather than k, becaues this
        # method can apparently obtain better convergence if a few extra vectors
        # are computed, then truncated to k.
        self.true_k, self.p = k, p
        StreamingPCA.__init__(self, d, p, B=B, Sparse=Sparse, Acc=Acc, X=X, xnorm2=xnorm2, num_acc=num_acc, Time=Time)

    def xnorm2_init(self, xnorm2):
        '''
        Because SPM initializes more vectors than it needs, we rewrite
        xnorm2_init to correctly calculate the accuracy with the true k
        value given.
        '''
        # Make sure xnorm2 is initalized and take the first accuracy reading
        self.xnorm2 = xnorm2
        if self.islist:
            # If the squared frobenius of X norm is not provided, calculate it yourself
            if xnorm2 is None:
                if self.Sparse:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=True)
                else:
                    self.xnorm2 = list_xnorm2(self.X, Sparse=False)
            # Take an initial reading of the explained variance. Note we
            # truncate to the top k vectors
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
        Input:
            final_sample: Boolean that indicates whether this accuracy is the final accuracy to be obtained or not.
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
        S = 1 / self.B * self.Xi.T @ (self.Xi @ self.Q)
        self.Q = la.qr(S, mode='economic')[0]

    def sparse_update(self):
        S = 1 / self.B * self.Xi.T.dot(self.Xi.dot(self.Q))
        self.Q = la.qr(S, mode='economic')[0]

class PM_mom(StreamingPCA):
    '''
    Implements the Mini-batch Power Method with Momentum found in "Accelerated
        Stochastic Power Iteration" by De Sa, He, Mitliagkas, Re, and Xu.
    '''
    def __init__(self, *args, beta_update=10, beta_scales = np.array([2/3, 0.99, 1, 1.01, 1.5]), **kwargs):
        '''
        beta_update: optional int > 0, the number of block iterations before
            beta is updated.
        beta_scales = optional 1D numpy array of scales to try for updating beta. Note
            that the case where beta is kept constant--1--will be added to
            beta_scales if it is not already included. Default given by
            Algorithm 3, Best Heavy Ball, in the reference paper.
        '''
        super().__init__(self, *args, **kwargs)
        self.beta_update = beta_update

        # Make sure beta_scales contains a 1.
        one_index = np.where(beta_scales==1)[0]
        if one_index.size == 0:
            self.beta_scales = np.hstack((1, beta_scales))
            self.one_index = 0
        else:
            self.beta_scales = beta_scales
            self.one_index = one_index[0]
        self.num_beta = beta_scales.size

        self.Q_vals = self.Q * np.ones((self.num_beta, self.d, self.k))
        self.R_vals = np.eye(self.k) * np.ones((self.num_beta, self.k, self.k))
        self.Q0_vals = np.zeros((self.num_beta, self.d, self.k))\

        self.rayleigh = np.zeros(self.num_beta)

    def add_block(self, Xi, final_sample=False):
        self.Xi = Xi
        # Initialize beta given the first block
        if self.sample_num == 0:
            self.beta = la.norm(self.Xi.dot(self.Q), ord='fro')**4 / 2
            self.beta_vals = self.beta * self.beta_scales

        StreamingPCA.add_block(self, Xi, final_sample=final_sample)

        if final_sample or self.block_num % self.beta_update == 0:
        # After self.beta_update iterations, update the beta value
            self.choose_beta()

    def choose_beta(self):
        # Look at doing this with broadcasting
        for i in range(self.num_beta):
            self.rayleigh[i] = la.norm(self.Xi.dot(self.Q_vals[i]), ord='fro')**2
        best_beta_index = np.argmax(self.rayleigh)
        self.beta = self.beta_vals[best_beta_index]

        # Update all values to the current best case values
        self.Q_vals[:] = self.Q_vals[best_beta_index]
        self.Q0_vals[:] = self.Q0_vals[best_beta_index]
        self.R_vals[:] = self.R_vals[best_beta_index]
        self.Q = self.Q_vals[best_beta_index]

        # Set a new set of beta_vals based on the updated beta
        self.beta_vals = self.beta * self.beta_scales


    def dense_update(self):
        # Look into doing this with broadcasting
        for i in range(self.num_beta):
            S = 1 / self.B * self.Xi.T @ (self.Xi @ self.Q_vals[i]) - self.beta_vals[i] * self.Q0_vals[i] @ la.inv(self.R_vals[i])
            self.Q0_vals[i] = self.Q_vals[i]
            self.Q_vals[i], self.R_vals[i] = la.qr(S, mode='economic')
        # Set Q to be the result achieved by our current Q
        self.Q = self.Q_vals[self.one_index]

    def sparse_update(self):
        for i in range(self.num_beta):
            S = 1 / self.B * self.Xi.T.dot(self.Xi.dot(self.Q_vals[i])) - self.beta_vals[i] * self.Q0_vals[i].dot(la.inv(self.R_vals[i]))
            self.Q0_vals[i] = self.Q_vals[i]
            self.Q_vals[i], self.R_vals[i] = la.qr(S, mode='economic')
        # Set Q to be the result achieved by our current Q
        self.Q = self.Q_vals[self.one_index]

class ADAM(StreamingPCA):
    def __init__(self, *args, eta=1e-3, beta_1 = 0.9, beta_2 = 0.999, delta=1e-8, unorm=2, bias_correction=False, b0_dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta, self.beta_1, self.beta_2, self.delta, self.unorm, self.bias_correction, self.b0_dim = eta, beta_1, beta_2, delta, unorm, bias_correction, b0_dim

        if self.b0_dim==0:
            self.b0 = 0
        elif self.b0_dim ==1:
            self.b0 = np.zeros(self.k)
        elif self.b0_dim == 2:
            self.b0 = np.zeros((self.d, self.k))
        else:
            raise ValueError("b0_dim options are 0: constant, 1: vector, or 2: matrix")

        self.m0 = np.zeros((self.d, self.k))
        self.stepvals = []

    def dense_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T @ (self.Xi @ self.Q) / self.B
        self.m0 = (self.beta_1 * self.m0 + (1 - self.beta_1) * G)

        if self.b0_dim==0:
            self.b0 = (self.beta_2 * self.b0 + (1 - self.beta_2) * np.linalg.norm(G, ord=self.unorm)**2)
        if self.b0_dim ==1:
            self.b0 = (self.beta_2 * self.b0 + (1 - self.beta_2) * np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        if self.b0_dim == 2:
            self.b0 = self.beta_2 * self.b0 + (1 - self.beta_2) * G**2

        if self.bias_correction:
            self.m0 /= (1 - self.beta_1**self.sample_num)
            self.b0 /= (1 - self.beta_2**self.sample_num)
        self.stepvals.append(self.eta/(np.sqrt(self.b0) + self.delta))
        self.Q += self.eta / (np.sqrt(self.b0) + self.delta) * self.m0
        self.Q = la.qr(self.Q, mode='economic')[0]

    def sparse_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
        self.m0 = (self.beta_1 * self.m0 + (1 - self.beta_1) * G)
        if self.b0_dim==0:
            self.b0 = (self.beta_2 * self.b0 + (1 - self.beta_2) * np.linalg.norm(G, ord=self.unorm)**2)
        if self.b0_dim ==1:
            self.b0 = (self.beta_2 * self.b0 + (1 - self.beta_2) * np.linalg.norm(G, ord=self.unorm, axis=0)**2)
        if self.b0_dim == 2:
            self.b0 = self.beta_2 * self.b0 + (1 - self.beta_2) * G**2
        if self.bias_correction:
            self.m0 /= (1 - self.beta_1**self.sample_num)
            self.b0 /= (1 - self.beta_2**self.sample_num)
        self.stepvals.append(self.eta/(np.sqrt(self.b0)+self.delta))
        self.Q += self.eta / (np.sqrt(self.b0) + self.delta) * self.m0
        self.Q = la.qr(self.Q, mode='economic')[0]

class RMSProp(StreamingPCA):
    def __init__(self, *args, gamma=.9, eta=1e-3, b0=1e-5, unorm=2, b0_dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma, self.eta, self.unorm, self.b0_dim = gamma, eta, unorm, b0_dim

        if self.b0_dim==0:
            self.b0 = b0
        elif self.b0_dim ==1:
            self.b0 = np.ones(self.k) * b0
        elif self.b0_dim == 2:
            self.b0 = np.ones((self.d, self.k)) * b0
        else:
            raise ValueError("b0_dim options are 0: constant, 1: vector, or 2: matrix")
        self.stepvals = []


    def dense_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T @ (self.Xi @ self.Q) / self.B
        if self.b0_dim==0:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * np.linalg.norm(G, ord=self.unorm)**2
        if self.b0_dim ==1:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * np.linalg.norm(G, ord=self.unorm, axis=0)**2
        if self.b0_dim == 2:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * G**2

        self.Q += self.eta * G / np.sqrt(self.b0)
        self.Q = la.qr(self.Q, mode='economic')[0]

        self.stepvals.append(self.eta / np.sqrt(self.b0))

    def sparse_update(self):
        # Make a local variable for the current value of Q1
        #Q0 = np.copy(self.Q)

        G = self.Xi.T.dot(self.Xi.dot(self.Q)) / self.B
        if self.b0_dim==0:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * np.linalg.norm(G, ord=self.unorm)**2
        if self.b0_dim ==1:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * np.linalg.norm(G, ord=self.unorm, axis=0)**2
        if self.b0_dim == 2:
            self.b0 = self.gamma * self.b0 + (1 - self.gamma) * G**2
        self.Q += self.eta * G / np.sqrt(self.b0)
        self.Q = la.qr(self.Q, mode='economic')[0]

        self.stepvals.append(self.eta / np.sqrt(self.b0))
