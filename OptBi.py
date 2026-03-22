import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from optAlgo import PSO
import tensorflow as tf
from dpcnn import SeqQSPR
from rdkit import RDLogger
from prepare import Tokenize
from vae import CNNVAE, CNNEncoder, CNNDecoder
def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    return seed
seed = setup(233666)
RDLogger.DisableLog('rdApp.*')


sa = SeqQSPR(molProperty="SA", verbose=False)
sa.load()
logp = SeqQSPR(molProperty="LOGP", verbose=False)
logp.load()
tpsa = SeqQSPR(molProperty="TPSA", verbose=False)
tpsa.load()
qspr = {"LOGP": logp, "TPSA": tpsa}


class GenerateMolecule:
    def __init__(self,
                 target,
                 molProp,
                 group_size=100,):

        self.group_size = group_size

        self.saScorer = sa

        self.qsprs = []
        for mp in molProp:
            self.qsprs.append(qspr[mp])
        self.molProp = molProp

        self.vae = CNNVAE(verbose=False)
        self.vae.load()
        self.decoder = self.vae.decoder
        self.max_len = self.vae.max_len
        self.latent_dim = self.vae.latent_dim
        self.tokenizer = Tokenize()

        self.target = np.array(target, dtype=np.float32)

        self.pso = PSO(self.fitnessFunction, self.diversify, initialGroup=None, pop=self.group_size, dim=self.latent_dim)

    def reconstruct(self, decoded):
        reconstructed = decoded.tolist()
        _reconstructed_ = []
        for i in reconstructed:
            try:
                i = i[:i.index(0)]
                i = i + [0]*(self.max_len-len(i))
                _reconstructed_.append(i)
            except ValueError:
                _reconstructed_.append(i)
        return np.array(_reconstructed_, dtype=np.int32)

    def calProperty(self, reconstructed):
        """
        :param shape (group_size, latent_dim)
        :return: SA value, property vector
        """
        reconstructed = np.int32(reconstructed)
        sa = self.saScorer(reconstructed).numpy().squeeze()
        target = np.array([qspr(reconstructed).numpy().squeeze() for qspr in self.qsprs])
        return sa.squeeze(), target.squeeze()

    def verify(self, reconstructed):
        reconstructed = ["".join(self.tokenizer.detokenize(i.tolist())) for i in reconstructed]
        validity = []

        for i in reconstructed:
            mol = Chem.MolFromSmiles(i)
            if mol:
                validity.append(1)
            else:
                validity.append(1e20)

        validity = np.array(validity)
        return validity

    def diversify(self, X):
        decoded = self.decoder(X).numpy().squeeze().argmax(-1)
        reconstructed = self.reconstruct(decoded)
        reconstructed = ["".join(self.tokenizer.detokenize(i.tolist())) for i in reconstructed]

        return reconstructed

    def fitnessFunction(self, latents):
        latents.shape = (self.group_size, self.latent_dim)
        decoded = self.decoder(latents).numpy().squeeze().argmax(-1)
        reconstructed = self.reconstruct(decoded)
        validity = self.verify(reconstructed)
        sa, properties = self.calProperty(reconstructed)
        fitness = np.abs(properties.T / self.target - 1)
        fitness = sa * fitness.squeeze() * validity
        return fitness.squeeze()

    def run_optimization(self,  target, seed=0):
        molProp = self.molProp if type(self.molProp) is str else "-".join(self.molProp)
        self.pso.run(target, seed=seed, propertyName=molProp)


# MultiObjective
def molopt(seed=0, targets="2.5", prop="logp"):
    if not os.path.exists(f"results/pso/MULTI-OBJECTIVE/{prop.upper()}/{targets}-Seed99.npz"):
        while seed < 100:
            try:
                path = f"results/pso/MULTI-OBJECTIVE/{prop.upper()}/{targets}-Seed{seed}.npz"
                target = [float(i) for i in targets.split("-")]
                if not os.path.exists(path):
                    molGen = GenerateMolecule(target=target, molProp=[prop.upper()], group_size=100)
                    molGen.run_optimization(target, seed=seed)
                    del molGen
                    gc.collect()
                seed += 1
            except:
                molopt(seed, targets)


# if __name__ == "__main__":
molopt(seed=0, targets="1.0", prop="logp")
molopt(seed=0, targets="2.0", prop="logp")
molopt(seed=0, targets="3.0", prop="logp")
molopt(seed=0, targets="4.0", prop="logp")
molopt(seed=0, targets="20.0", prop="tpsa")
molopt(seed=0, targets="40.0", prop="tpsa")
molopt(seed=0, targets="60.0", prop="tpsa")
molopt(seed=0, targets="80.0", prop="tpsa")


