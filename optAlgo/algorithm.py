import os
import sys
import types
import random
import warnings
import numpy as np
import multiprocessing
from scipy import spatial
from tqdm import tqdm, trange
from functools import lru_cache
from abc import ABCMeta, abstractmethod
from types import MethodType, FunctionType
def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    return seed
seed = setup(233666)
if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')


def set_run_mode(func, mode):
    """
    :param func:
    :param mode: <string> can be common, vectorization , parallel, cached
    :return:
    """
    if mode == 'multiprocessing' and sys.platform == 'win32':
        warnings.warn('multiprocessing not support in windows, turning to multithreading')
        mode = 'multithreading'
    if mode == 'parallel':
        mode = 'multithreading'
        warnings.warn('use multithreading instead of parallel')
    func.__dict__['mode'] = mode
    return


def func_transformer(func):
    """
    - transform this kind of function

    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2

    into this kind of function:

    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2

    getting vectorial performance if possible:

    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    :param func:
    :return:
    """

    if (func.__class__ is FunctionType) and (func.__code__.co_argcount > 1):
        def func_transformed(X):
            return np.array([func(*tuple(x)) for x in X])
        return func_transformed
    if (func.__class__ is MethodType) and (func.__code__.co_argcount > 2):
        def func_transformed(X):
            return np.array([func(tuple(x)) for x in X])
        return func_transformed
    if getattr(func, 'is_vector', False):
        set_run_mode(func, 'vectorization')

    mode = getattr(func, 'mode', 'others')
    valid_mode = ('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached', 'others')
    assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)
    if mode == 'vectorization':
        return func
    elif mode == 'cached':
        @lru_cache(maxsize=None)
        def func_cached(x):
            return func(x)
        def func_warped(X):
            return np.array([func_cached(tuple(x)) for x in X])
        return func_warped
    elif mode == 'multithreading':
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool()
        def func_transformed(X):
            return np.array(pool.map(func, X))
        return func_transformed
    elif mode == 'multiprocessing':
        from multiprocessing import Pool
        pool = Pool()
        def func_transformed(X):
            return np.array(pool.map(func, X))
        return func_transformed
    else:  # common
        def func_transformed(X):
            return func(X)
        return func_transformed


class SkoBase(metaclass=ABCMeta):
    def register(self, operator_name, operator, *args, **kwargs):
        """
        regeister udf to the class
        :param operator_name: string
        :param operator: a function, operator itself
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        """

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)
        setattr(self, operator_name, types.MethodType(operator_wapper, self))
        return self

    def fit(self, *args, **kwargs):
        warnings.warn('.fit() will be deprecated in the future. use .run() instead.' , DeprecationWarning)
        return self.run(*args, **kwargs)


class PSO(SkoBase):
    """
    Do PSO (Particle swarm optimization) algorithm.
    This algorithm was adapted from the earlier works of J. Kennedy and R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.
    The position update can be defined as:
    .. math::
       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using the computed velocity at :math:`t+1`.
    Furthermore, the velocity update is defined as:
    .. math::
       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)] + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters respectively.
    They control the particle's behavior given two choices:
    (1) to follow its *personal best*
    (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : The func you want to do optimal
    max_iter : Max of iter iterations
    lb : array_like The lower bound of every variables of func
    ub : array_like The upper bound of every variables of func
    constraint_ueq : tuple unequal constraint
    ----------------------
    NOTES:
    X, Y >> Current particles of the group.
    pbestX, pbestY >> Best particles of the group.
    gbestX, gbestY >> Global best particles.

    Examples
    -----------------------------
    see https://scikit-opt.github.io/scikit-opt/#/en/README?id=_3-psoparticle-swarm-optimization
    """

    def __init__(self, func1, func2, initialGroup, pop=None, dim=None, max_iter=1000):

        self.func1 = func_transformer(func1)
        self.func2 = func_transformer(func2)
        if initialGroup:
            self.pop = initialGroup.shape[0]  # number of particles
            self.n_dim = initialGroup.shape[1]  # dimension of particles, which is the number of variables of func
            self.max_iter = max_iter  # max iteration steps

            self.lb, self.ub = 1e4*initialGroup.min(0), 1e4*initialGroup.max(0)
            assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
            assert np.all(np.greater(self.ub, self.lb)), 'upper-bound must be greater than lower-bound'

            self.X = initialGroup
        else:
            self.pop = pop  # number of particles
            self.n_dim = dim  # dimension of particles, which is the number of variables of func

            initialGroup = np.random.randn(self.pop, self.n_dim)
            initialGroup = np.float32(initialGroup)
            X = self.func2(initialGroup)
            duplicates = [X.count(i) for i in X]
            while duplicates.count(1) < len(duplicates):
                first = 0
                for i in range(len(X)):
                    if duplicates[i] != 1 and first != 0:
                        X[i] = np.random.randn(1, self.n_dim)
                    elif duplicates[i] != 1 and first == 0:
                        first += 1
                    else:
                        pass
            self.X = initialGroup
            self.max_iter = max_iter  # max iteration steps

            self.lb, self.ub = 1e4*initialGroup.min(0), 1e4*initialGroup.max(0)
            assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
            assert np.all(np.greater(self.ub, self.lb)), 'upper-bound must be greater than lower-bound'

        self.recorderHIST = {'X': [], 'Y': []}

        # Parameter initialization of the initialized group of particles.
        self.Y = self.func1(self.X)  # The fitness value of the initialized group of particles.
        self.V = np.random.randn(*self.X.shape)  # The speed of the initialized group of particles.
        self.update_C(0)
        self.pbest_x = self.X.copy()  # The initialization of personal best location of every particle in history.
        self.pbest_y = self.Y.copy()  # The initialization of personal best fitness value of every particle in history.
        self.gbest_x = self.X[np.equal(self.Y, self.Y.squeeze().min())]  # global best location for all particles.
        self.gbest_y = self.Y.squeeze().min()  # global best y for all particles
        self.update_W()
        self.recorder()

    def update_W(self):
        min_y = self.pbest_y.min()
        max_y = self.pbest_y.max()
        mean_y = self.pbest_y.mean()
        cond = np.less_equal(self.pbest_y.squeeze(), mean_y.squeeze())
        w = 0.4 + 0.5 * (self.pbest_y.squeeze() - min_y)/(max_y - mean_y + 1e-20)
        self.W = np.where(cond, w, 0.9)
        self.W.shape = (self.pop, 1)
        self.W = np.repeat(self.W, self.n_dim, 1)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.W * self.V
        self.V += self.cp * r1 * (self.pbest_x - self.X)
        self.V += self.cg * r2 * (self.gbest_x - self.X)

    def update_C(self, iter):
        # cp and cg are the learning factors of personal best and global best respectively
        # larger cp leads to more locally searching while larger cg leads to the early convergence to the local optimum.
        cpi, cpf, cgi, cgf = 2.0, 1.0, 1.5, 2.5
        self.cp = cpi + iter * (cpf - cpi) / self.max_iter
        self.cg = cgi + iter * (cgf - cgi) / self.max_iter

    def update_pbest(self):
        X = self.func2(self.X)
        pbestX = self.func2(self.pbest_x)
        diversity1 = []
        for i in pbestX:
            if pbestX.count(i) > 1:
                diversity1.append(True)
            else:
                diversity1.append(False)

        diversity1 = np.array(diversity1)
        diversity2 = np.array([i not in pbestX for i in X])
        diversity = np.logical_or(diversity1, diversity2)
        self.pbest_y = self.pbest_y
        self.need_update = np.less(self.Y, self.pbest_y)
        self.need_update = np.logical_and(diversity, self.need_update)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)
        self.need_update.shape = (self.pop, 1)
        self.mask = np.repeat(self.need_update, self.n_dim, axis=1)
        self.pbest_x = np.where(self.mask, self.X, self.pbest_x)

    def update_gbest(self):
        idx_min = self.pbest_y.argmin()
        if self.pbest_y.min() < self.gbest_y:
            self.gbest_x = self.pbest_x[idx_min].copy()
            self.gbest_y = self.pbest_y.squeeze().min()

    def recorder(self):
        self.recorderHIST['X'].append(self.pbest_x)
        self.recorderHIST['Y'].append(self.pbest_y)

    def run(self, target, seed=0, propertyName=None, multiobjective=True, objective=None):
        """
        If precision is None, it will run the number of max_iter steps
        If precision is a float, the loop will stop if continuous N difference between **pbest** less than precision
        """
        tol = 0.5 * (len(target) + int(multiobjective))
        setup(seed)
        memo = self.gbest_y
        for step in trange(1, self.max_iter + 1):
            self.update_X()
            self.Y = self.func1(self.X)
            self.update_pbest()
            self.update_gbest()
            self.update_C(step)
            self.update_W()
            self.update_V()
            if self.gbest_y != memo:
                self.recorder()
                memo = self.gbest_y
            print(f'Iter: {step}, Best fit of this epoch: {round(self.pbest_y.min().tolist(), 4)}, Std: '
                  f'{round(np.std(self.pbest_y), 4)}, Accomplished:{np.less(self.pbest_y, tol).sum()/self.pop}')
            if np.less(self.pbest_y, tol).sum() == self.pop:
                break
        self.recorder()

        print("ITERATIONS ARE FINISHED. NOW LOGGING OPTIMIZATION INFORMATION.")
        multiobjective = objective if objective else ("MULTI-OBJECTIVE" if multiobjective else "UNI-OBJECTIVE")
        target = "-".join([str(i) for i in target])
        path = f"results/pso/{multiobjective}/{propertyName}/{target}-Seed{seed}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, HistX=self.recorderHIST["X"], HistY=self.recorderHIST["Y"])

