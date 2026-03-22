import os
import gc
import argparse
import json
import random
import warnings
from zipfile import BadZipFile
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
from config import PSO_RESULTS_DIR, set_working_directory

# Ensure all relative model/data paths resolve from project root.
set_working_directory()


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


def log_info(message):
    print(f"[OptTri] {message}")


def release_seed_memory(seed, path):
    collected = gc.collect()
    log_info(f"Seed={seed} cleanup done (gc_collected={collected}) path={os.path.abspath(path)}")


def checkpoint_file(objective, targets):
    ckpt_dir = os.path.join(PSO_RESULTS_DIR, objective, "LOGP-TPSA", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{targets}.checkpoint.json")


def load_checkpoint(ckpt_path, fallback_seed):
    if not os.path.exists(ckpt_path):
        return fallback_seed
    try:
        with open(ckpt_path, encoding="utf-8") as reader:
            state = json.load(reader)
        return int(state.get("next_seed", fallback_seed))
    except Exception:
        return fallback_seed


def save_checkpoint(ckpt_path, state):
    with open(ckpt_path, "w", encoding="utf-8") as writer:
        json.dump(state, writer, ensure_ascii=True, indent=2)


def seed_file_is_healthy(npz_path):
    if not os.path.exists(npz_path):
        return False
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if "HistX" not in data or "HistY" not in data:
                return False
            _ = data["HistX"][-1]
            _ = data["HistY"][-1]
        return True
    except (BadZipFile, EOFError, OSError, ValueError, KeyError, IndexError):
        return False


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
        fitness = fitness.sum(-1)
        fitness = sa * fitness.squeeze() * validity
        return fitness.squeeze()

    def run_optimization(self,  target, seed=0, objective="MULTI-OBJECTIVE"):
        molProp = self.molProp if type(self.molProp) is str else "-".join(self.molProp)
        self.pso.run(target, seed=seed, propertyName=molProp, objective=objective)


# MultiObjective
def molopt(seed=0, targets="2.0-20.0", objective="MULTI-OBJECTIVE", seed_end=100):
    single_seed_backfill = (seed_end == seed + 1)
    ckpt_path = checkpoint_file(objective, targets)
    if not single_seed_backfill:
        resume_seed = load_checkpoint(ckpt_path, seed)
        if resume_seed > seed:
            log_info(f"Resume from checkpoint: seed={resume_seed}, file={os.path.abspath(ckpt_path)}")
        seed = resume_seed

    final_seed = seed_end - 1
    final_path = os.path.join(PSO_RESULTS_DIR, objective, "LOGP-TPSA", f"{targets}-Seed{final_seed}.npz")
    if not seed_file_is_healthy(final_path):
        log_info(f"Start target={targets}, objective={objective}, seeds={seed}-{seed_end - 1}")
        while seed < seed_end:
            molGen = None
            seed_ok = False
            path = os.path.join(PSO_RESULTS_DIR, objective, "LOGP-TPSA", f"{targets}-Seed{seed}.npz")
            target = [float(i) for i in targets.split("-")]
            try:
                if not seed_file_is_healthy(path):
                    if os.path.exists(path):
                        os.remove(path)
                    molGen = GenerateMolecule(target=target, molProp=["LOGP", "TPSA"], group_size=100)
                    molGen.run_optimization(target, seed=seed, objective=objective)
                    log_info(f"Generated seed={seed} for {objective}/LOGP-TPSA/{targets}")
                    log_info(f"Saved seed output: {os.path.abspath(path)}.npz")
                    seed_ok = seed_file_is_healthy(path)
                else:
                    log_info(f"Seed={seed} already exists at: {os.path.abspath(path)}")
                    seed_ok = True
            except Exception as e:
                log_info(f"Seed={seed} failed at {os.path.abspath(path)} with error: {e}")
            finally:
                if molGen is not None:
                    del molGen
                if not single_seed_backfill:
                    save_checkpoint(ckpt_path, {
                        "objective": objective,
                        "property": "LOGP-TPSA",
                        "target": targets,
                        "last_done_seed": seed,
                        "next_seed": seed + 1,
                        "seed_end": seed_end,
                        "last_seed_ok": bool(seed_ok),
                    })
                release_seed_memory(seed, path)
                seed += 1
        if not single_seed_backfill:
            save_checkpoint(ckpt_path, {
                "objective": objective,
                "property": "LOGP-TPSA",
                "target": targets,
                "last_done_seed": seed_end - 1,
                "next_seed": seed_end,
                "seed_end": seed_end,
                "status": "completed",
            })
    else:
        log_info(f"Skip target={targets}, objective={objective} (already complete)")


def run_chunk(chunk_idx, chunk_size=100, with_stats=True):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    objective = f"MULTI-OBJECTIVE-R{chunk_idx}"
    chunk_seed_start = (chunk_idx - 1) * chunk_size
    chunk_seed_end = chunk_idx * chunk_size
    log_info(f"Chunk {chunk_idx} seed window: {chunk_seed_start}-{chunk_seed_end - 1}")
    targets = ["1.0-20.0", "2.0-40.0", "3.0-60.0", "4.0-80.0"]
    for target in targets:
        molopt(seed=chunk_seed_start, targets=target, objective=objective, seed_end=chunk_seed_end)

    if with_stats:
        from model_stats import psoValidity, psoGeneratedDistribution, psoDescription
        for target in targets:
            psoValidity(objective=objective, prop="LOGP-TPSA", prefix=target,
                        seed_start=chunk_seed_start, n_seeds=chunk_size)
            psoGeneratedDistribution(objective=objective, prop="LOGP-TPSA", prefix=target)
            psoDescription(objective=objective, prop="LOGP-TPSA", prefix=target)
        log_info(f"Statistics finished for {objective}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run TRI-objective PSO experiments in chunks.")
    parser.add_argument("--total-runs", type=int, default=500, help="Total runs per target.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Runs per chunk.")
    parser.add_argument("--with-stats", action="store_true", help="Run model_stats after each chunk.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.total_runs % args.chunk_size != 0:
        raise ValueError("total-runs must be divisible by chunk-size")

    n_chunks = args.total_runs // args.chunk_size
    log_info(f"Run plan: total_runs={args.total_runs}, chunk_size={args.chunk_size}, chunks={n_chunks}")
    for i in range(1, n_chunks + 1):
        log_info(f"==== Start chunk {i}/{n_chunks} ====")
        run_chunk(chunk_idx=i, chunk_size=args.chunk_size, with_stats=args.with_stats)
        log_info(f"==== Finished chunk {i}/{n_chunks} ====")
