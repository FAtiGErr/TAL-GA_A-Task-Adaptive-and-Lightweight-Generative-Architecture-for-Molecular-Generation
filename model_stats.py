import os
import csv
import glob
import argparse
import random
import re
from zipfile import BadZipFile
import numpy as np
from os import path
import pandas as pd
from rdkit import Chem
from vae import CNNVAE
import tensorflow as tf
from dpcnn import SeqQSPR
from rdkit.Chem import Draw
from prepare import Tokenize
from tqdm import tqdm, trange
from prepare.synava import SaScore
from scipy.spatial.distance import cosine
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from config import (
    TOKEN_SET_DIR,
    MOSES_DS_DIR,
    RESULTS_MODELS_QSPR_DIR,
    RESULTS_MODELS_VAE_DIR,
    RESULTS_MODELS_VAE_WRONG_DIR,
    RESULTS_FEATURE_DIR,
    EMBEDDING_WEIGHTS_DIR,
    MOL_PROPERTIES_DIR,
    PSO_RESULTS_DIR,
    set_working_directory,
)


# Ensure relative paths behave consistently across devices/runners.
set_working_directory()


UNI_TASKS = [
    ("LOGP", "1.0"), ("LOGP", "2.0"), ("LOGP", "3.0"), ("LOGP", "4.0"),
    ("TPSA", "20.0"), ("TPSA", "40.0"), ("TPSA", "60.0"), ("TPSA", "80.0"),
]
TRI_TASKS = [("LOGP-TPSA", "1.0-20.0"), ("LOGP-TPSA", "2.0-40.0"), ("LOGP-TPSA", "3.0-60.0"), ("LOGP-TPSA", "4.0-80.0")]


def makePredictions():
    tokenizer = Tokenize(dictPath=os.path.join(TOKEN_SET_DIR, "Unigram.csv"),
                         tokenPath=os.path.join(TOKEN_SET_DIR, "tokenList.csv"))
    qsprs = [
        SeqQSPR(molProperty="LOGP"),
        SeqQSPR(molProperty="TPSA"),
        SeqQSPR(molProperty="SA"),
    ]

    for i in qsprs:
        i.load()

    maxlen = qsprs[0].max_len

    test_file = open(os.path.join(MOSES_DS_DIR, "test.csv"), encoding="utf-8")
    csvfile = open(os.path.join(MOSES_DS_DIR, "testPrediction.csv"), "w", encoding='utf-8', newline="")
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow("SMILES,LOGP,TPSA,SA,PredLOGP,PredTPSA,PredSA".split(","))
    for idx, line in enumerate(tqdm(test_file)):
        if 0 < idx:
            try:
                line = line.strip()
                arr = tokenizer.tokenize(line)
                seqlen = len(arr)
                if seqlen <= maxlen:
                    arr = [int(tokenizer.t2i[i]) for i in arr]
                    arr += (maxlen - seqlen) * [0]
                    arr = np.array([arr])
                    mol = Chem.MolFromSmiles(line)
                    calculate_list = ["MolLogP", "TPSA"]
                    SA = SaScore()
                    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(calculate_list)
                    logp, tpsa = calculator.CalcDescriptors(mol)
                    sa = SA(mol)
                    preds = [round(qspr(arr).numpy().squeeze().tolist(), 4) for qspr in qsprs]
                    outputs = [line, round(logp, 4), round(tpsa, 4), round(sa, 4)] + preds
                    formated_outputs = "{}, LogP: {}, TPSA: {}, SA: {}, _LogP_: {}, _TPSA_: {}, _SA_: {}".format(*outputs)
                    print(formated_outputs)
                    csv_writer.writerow(outputs)
            except:
                pass
    csvfile.close()
    test_file.close()


def statistics():
    with open(os.path.join(MOSES_DS_DIR, "testPrediction.csv"), encoding="utf-8") as csvreader:
        SMILES = []
        LOGP = []
        TPSA = []
        SA = []
        PredLOGP = []
        PredTPSA = []
        PredSA = []
        for idx, line in enumerate(csvreader):
            if idx != 0:
                smi, logp, tpsa, sa, predlogp, predtpsa, predsa = line.split(",")
                SMILES.append(smi)
                LOGP.append(float(logp))
                TPSA.append(float(tpsa))
                SA.append(float(sa))
                PredLOGP.append(float(predlogp))
                PredTPSA.append(float(predtpsa))
                PredSA.append(float(predsa))

    logpMAE = mean_absolute_error(LOGP, PredLOGP)
    tpsaMAE = mean_absolute_error(TPSA, PredTPSA)
    saMAE = mean_absolute_error(SA, PredSA)

    logpRMSE = mean_squared_error(LOGP, PredLOGP)**0.5
    tpsaRMSE = mean_squared_error(TPSA, PredTPSA)**0.5
    saRMSE = mean_squared_error(SA, PredSA)**0.5

    logpR2 = r2_score(LOGP, PredLOGP)
    tpsaR2 = r2_score(TPSA, PredTPSA)
    saR2 = r2_score(SA, PredSA)

    os.makedirs(RESULTS_MODELS_QSPR_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_MODELS_QSPR_DIR, "performance.txt"), "wb") as csvwriter:
        csvwriter.write("Metrics,LOGP,TPSA,SA\n".encode("utf-8"))
        csvwriter.write(f"MAE,{round(logpMAE,4)},{round(tpsaMAE,4)},{round(saMAE,4)}\n".encode("utf-8"))
        csvwriter.write(f"RMSE,{round(logpRMSE,4)},{round(tpsaRMSE,4)},{round(saRMSE,4)}\n".encode("utf-8"))
        csvwriter.write(f"R2,{round(logpR2,4)},{round(tpsaR2,4)},{round(saR2,4)}\n".encode("utf-8"))


def VAE_Validity():
    mosesTest = []
    with open(os.path.join(MOSES_DS_DIR, "test.csv")) as reader:
        for _, i in enumerate(reader):
            if _ != 0:
                mosesTest.append(i.strip())
    tokenizer = Tokenize()
    vae = CNNVAE()
    vae.load()
    enc = vae.encoder
    dec = vae.decoder
    raw = []
    smis = []
    for line in mosesTest:
        raw_i = tokenizer.tokenize(line)
        try:
            line = [int(tokenizer.t2i[i]) for i in raw_i]
            seqlen = len(line)
            line += (vae.max_len - seqlen) * [0]
            line = np.array([line], dtype=np.int32)
            latent, mu, sigma = enc(line)
            decoded = dec(latent).numpy().squeeze().argmax(-1)
            rec = "".join(tokenizer.detokenize(decoded.tolist()))
            smis.append(rec)
            raw_i = "".join(raw_i)
            raw.append(raw_i)
            print(f"RAW: {raw_i}")
            print(f"REC: {rec}")
            print(f"Same?: {raw_i==rec}\n")
        except KeyError:
            pass
    validity = []
    for i in trange(len(smis)):
        if raw[i] == smis[i]:
            validity.append(True)
        else:
            validity.append(False)

    os.makedirs(RESULTS_MODELS_VAE_DIR, exist_ok=True)
    os.makedirs(RESULTS_MODELS_VAE_WRONG_DIR, exist_ok=True)
    df = pd.DataFrame({"Raw": raw, "Recon": smis, "Validity": validity})
    df.to_csv(os.path.join(RESULTS_MODELS_VAE_DIR, "Validity.csv"), index=False)
    print(f"The validity is : {np.array(validity).sum()/len(mosesTest)}")  # 97.44%  176074
    df1 = df[df["Validity"] == False]
    del df1["Validity"]
    df1.index = range(len(df1))
    df1.to_csv(os.path.join(RESULTS_MODELS_VAE_DIR, "Wrongly Decoded.csv"))
    idx = []
    for i in range(len(df1)):
        try:
            img = Draw.MolToImage(Chem.MolFromSmiles(df1['Recon'][i]))
            img.save(os.path.join(RESULTS_MODELS_VAE_WRONG_DIR, f"{i}Rec.bmp"))
            img = Draw.MolToImage(Chem.MolFromSmiles(df1['Raw'][i]))
            img.save(os.path.join(RESULTS_MODELS_VAE_WRONG_DIR, f"{i}Raw.bmp"))
            idx.append(i)
        except:
            pass
    df1.loc[idx].to_csv(os.path.join(RESULTS_MODELS_VAE_WRONG_DIR, "Raw-Rec SMILES.csv"), index=False)



def token_similarity(embedding_type="Word2vecEmbedding"):
    tokens = ["C(=O)N", "CN", "C(=O)", "CO", "CCN", "C(=O)O", "COC", "C(C)C", "c1ccccc1", "C(F)(F)F"]
    w2v_embedding_file = os.path.join(EMBEDDING_WEIGHTS_DIR, "Word2vecEmbedding.csv")
    w2v_dataframe = pd.read_csv(w2v_embedding_file)
    token_list = w2v_dataframe["Token"].values.tolist()
    embedding_file = os.path.join(EMBEDDING_WEIGHTS_DIR, embedding_type + ".csv")
    if os.path.normpath(embedding_file) == os.path.normpath(os.path.join(EMBEDDING_WEIGHTS_DIR, "Word2vecEmbedding.csv")):
        token_idx = [token_list.index(i) for i in tokens]
        vectors = pd.read_csv(embedding_file).values[:, 1:]
    else:
        token_idx = [token_list.index(i) + 1 for i in tokens]
        vectors = pd.read_csv(embedding_file).values
    token_vecs = vectors[token_idx]
    mtx = {}
    for i in range(len(token_vecs)):
        _ = []
        for j in range(len(token_vecs)):
            _.append(round(1 - cosine(token_vecs[i].tolist(), token_vecs[j].tolist()), 3))
        mtx[tokens[i]] = _
    os.makedirs(RESULTS_FEATURE_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_FEATURE_DIR, embedding_type + "_Similarity.csv")
    pd.DataFrame(mtx, index=tokens).to_csv(save_path)


def data_description():
    logp_train = os.path.join(MOL_PROPERTIES_DIR, "LOGP", "train.csv")
    logp_test = os.path.join(MOL_PROPERTIES_DIR, "LOGP", "test.csv")
    tpsa_train = os.path.join(MOL_PROPERTIES_DIR, "TPSA", "train.csv")
    tpsa_test = os.path.join(MOL_PROPERTIES_DIR, "TPSA", "test.csv")
    sa_train = os.path.join(MOL_PROPERTIES_DIR, "SA", "train.csv")
    sa_test = os.path.join(MOL_PROPERTIES_DIR, "SA", "test.csv")

    indices = ["Samples", "Mean", "Median", "LowerQuantile", "UpperQuantile", "StandardDeviation", "Skewness", "Kurtosis"]

    logp_train_statistics = []
    logp_test_statistics = []
    tpsa_train_statistics = []
    tpsa_test_statistics = []
    sa_train_statistics = []
    sa_test_statistics = []

    logp = pd.read_csv(logp_train)
    logp.columns = ["SMILES", "Value"]
    logp_train_statistics.append(len(logp))
    logp_train_statistics.append(logp["Value"].mean())
    logp_train_statistics.append(logp["Value"].median())
    logp_train_statistics.append(logp["Value"].quantile(q=0.25))
    logp_train_statistics.append(logp["Value"].quantile(q=0.75))
    logp_train_statistics.append(logp["Value"].std())
    logp_train_statistics.append(logp["Value"].skew())
    logp_train_statistics.append(logp["Value"].kurt())
    del logp

    logp = pd.read_csv(logp_test)
    logp.columns = ["SMILES", "Value"]
    logp_test_statistics.append(len(logp))
    logp_test_statistics.append(logp["Value"].mean())
    logp_test_statistics.append(logp["Value"].median())
    logp_test_statistics.append(logp["Value"].quantile(q=0.25))
    logp_test_statistics.append(logp["Value"].quantile(q=0.75))
    logp_test_statistics.append(logp["Value"].std())
    logp_test_statistics.append(logp["Value"].skew())
    logp_test_statistics.append(logp["Value"].kurt())
    del logp

    tpsa = pd.read_csv(tpsa_train)
    tpsa.columns = ["SMILES", "Value"]
    tpsa_train_statistics.append(len(tpsa))
    tpsa_train_statistics.append(tpsa["Value"].mean())
    tpsa_train_statistics.append(tpsa["Value"].median())
    tpsa_train_statistics.append(tpsa["Value"].quantile(q=0.25))
    tpsa_train_statistics.append(tpsa["Value"].quantile(q=0.75))
    tpsa_train_statistics.append(tpsa["Value"].std())
    tpsa_train_statistics.append(tpsa["Value"].skew())
    tpsa_train_statistics.append(tpsa["Value"].kurt())
    del tpsa

    tpsa = pd.read_csv(tpsa_test)
    tpsa.columns = ["SMILES", "Value"]
    tpsa_test_statistics.append(len(tpsa))
    tpsa_test_statistics.append(tpsa["Value"].mean())
    tpsa_test_statistics.append(tpsa["Value"].median())
    tpsa_test_statistics.append(tpsa["Value"].quantile(q=0.25))
    tpsa_test_statistics.append(tpsa["Value"].quantile(q=0.75))
    tpsa_test_statistics.append(tpsa["Value"].std())
    tpsa_test_statistics.append(tpsa["Value"].skew())
    tpsa_test_statistics.append(tpsa["Value"].kurt())
    del tpsa

    sa = pd.read_csv(sa_train)
    sa.columns = ["SMILES", "Value"]
    sa_train_statistics.append(len(sa))
    sa_train_statistics.append(sa["Value"].mean())
    sa_train_statistics.append(sa["Value"].median())
    sa_train_statistics.append(sa["Value"].quantile(q=0.25))
    sa_train_statistics.append(sa["Value"].quantile(q=0.75))
    sa_train_statistics.append(sa["Value"].std())
    sa_train_statistics.append(sa["Value"].skew())
    sa_train_statistics.append(sa["Value"].kurt())
    del sa

    sa = pd.read_csv(sa_test)
    sa.columns = ["SMILES", "Value"]
    sa_test_statistics.append(len(sa))
    sa_test_statistics.append(sa["Value"].mean())
    sa_test_statistics.append(sa["Value"].median())
    sa_test_statistics.append(sa["Value"].quantile(q=0.25))
    sa_test_statistics.append(sa["Value"].quantile(q=0.75))
    sa_test_statistics.append(sa["Value"].std())
    sa_test_statistics.append(sa["Value"].skew())
    sa_test_statistics.append(sa["Value"].kurt())
    del sa

    pd.DataFrame({"logp_train": logp_train_statistics,
                  "logp_test":logp_test_statistics,
                  "tpsa_train": tpsa_train_statistics,
                  "tpsa_test": tpsa_test_statistics,
                  "sa_train": sa_train_statistics,
                  "sa_test": sa_test_statistics},
                 index=indices).to_csv(os.path.join(MOL_PROPERTIES_DIR, "description.csv"), float_format="%.3f")


def psoValidity(objective="MULTI-OBJECTIVE", prop="LOGP", prefix="2.5-25.0", seed_start=0, n_seeds=100,
                seed_list=None, strict=False):
    tokenizer = Tokenize()
    vae = CNNVAE()
    vae.load()
    dec = vae.decoder

    smis = []
    seeds = list(seed_list) if seed_list is not None else list(range(seed_start, seed_start + n_seeds))
    loaded = 0
    corrupted = []
    for s in tqdm(seeds, desc=f"Validity {objective}/{prop}/{prefix}"):
        npz_file = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix}-Seed{s}.npz")
        if not os.path.exists(npz_file):
            if strict:
                raise FileNotFoundError(npz_file)
            continue
        try:
            data = np.load(npz_file, allow_pickle=False)
            _ = data["HistX"][-1]
        except (BadZipFile, EOFError, OSError, ValueError, KeyError, IndexError):
            corrupted.append(s)
            if strict:
                raise
            continue
        loaded += 1
        for i in data["HistX"][-1]:
            latent = i
            latent.shape = 1, 200
            decoded = dec(latent).numpy().squeeze().argmax(-1).tolist()
            smis.append("".join(tokenizer.detokenize(decoded)))

    if len(smis) == 0:
        raise RuntimeError(f"No particles decoded for {objective}/{prop}/{prefix}. Checked seeds: {len(seeds)}")

    validity = []
    for i in smis:
        mol = Chem.MolFromSmiles(i)
        if mol:
            validity.append(1)
        else:
            validity.append(0)

    df = pd.DataFrame({"SMILES": smis, "Validity": validity})
    out_validity = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix} Validity.csv")
    os.makedirs(os.path.dirname(out_validity), exist_ok=True)
    df.to_csv(out_validity, index=False)
    SMILES = df["SMILES"]
    SMILES_ = df["SMILES"].drop_duplicates()
    validity_ratio = np.array(validity).sum() * 100 / len(SMILES)
    diversity_ratio = len(SMILES_) / len(SMILES) * 100
    print(f"The validity is : {validity_ratio}%")
    print(f"The diversity is : {diversity_ratio}%")
    print(f"Loaded {loaded}/{len(seeds)} seed files for {objective}/{prop}/{prefix}")
    if corrupted:
        print(f"Skipped corrupted seeds for {objective}/{prop}/{prefix}: {corrupted}")
    return {
        "objective": objective,
        "property": prop,
        "target": prefix,
        "seed_total": len(seeds),
        "seed_loaded": loaded,
        "seed_corrupted": len(corrupted),
        "corrupted_seeds": " ".join(map(str, corrupted)),
        "decoded_smiles": len(df),
        "validity_percent": validity_ratio,
        "diversity_percent": diversity_ratio,
    }


def psoGeneratedDistribution(objective="MULTI-OBJECTIVE", prop="LOGP", prefix="2.5-25.0", force=False):
    property_path = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix} Property.csv")
    validity_path = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix} Validity.csv")
    if force or not os.path.exists(property_path):
        print(f"Dataframe {objective} {prop} {prefix} Loading.")
        df = pd.read_csv(validity_path)
        print(f"Dataframe {objective} {prop} {prefix} Loaded.")
        calculate_list = ["MolLogP", "TPSA"]
        calculator = MoleculeDescriptors.MolecularDescriptorCalculator(calculate_list)
        SA = SaScore()
        print("Calculating the metrics")
        mols = []
        idx = []
        for i in trange(len(df["SMILES"])):
            try:
                mol = Chem.MolFromSmiles(df["SMILES"][i])
                if mol is None:
                    continue
                mols.append(mol)
                idx += [i]
            except:
                pass
        if not idx:
            raise RuntimeError(f"No valid molecules left for property stats: {objective}/{prop}/{prefix}")
        df = df.loc[idx]
        logps = []
        for i in tqdm(mols):
            logps.append(calculator.CalcDescriptors(i)[0])
        df["LogP"] = logps
        tpsas = []
        for i in tqdm(mols):
            tpsas.append(calculator.CalcDescriptors(i)[1])
        df["TPSA"] = tpsas
        sas = []
        for i in tqdm(mols):
            sas.append(SA(i))
        df["SA"] = sas
        if "Unnamed: 0" in df.columns:
            del df["Unnamed: 0"]
        if "Validity" in df.columns:
            del df["Validity"]
        os.makedirs(os.path.dirname(property_path), exist_ok=True)
        df.to_csv(property_path, index=False)
        print("Property saved.\n")
    return property_path


def psoDescription(objective="MULTI-OBJECTIVE", prop="LOGP", prefix="2.5-25.0", force=False):
    df_path = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix} Property.csv")
    out_path = os.path.join(PSO_RESULTS_DIR, objective, prop, f"{prefix} description@10K.csv")

    if (not force) and os.path.exists(out_path):
        return out_path

    indices = ["Mean", "Median", "LowerQuantile", "UpperQuantile", "StandardDeviation", "Skewness", "Kurtosis", "Diversity"]

    logp_statistics = []
    tpsa_statistics = []
    sa_statistics = []

    df = pd.read_csv(df_path)
    if len(df) == 0:
        raise RuntimeError(f"No rows in property table: {df_path}")
    len_df = len(df)
    df.drop_duplicates(inplace=True)
    len_df1 = len(df)
    diversity = len_df1/len_df

    logp_statistics.append(df["LogP"].mean())
    logp_statistics.append(df["LogP"].median())
    logp_statistics.append(df["LogP"].quantile(q=0.25))
    logp_statistics.append(df["LogP"].quantile(q=0.75))
    logp_statistics.append(df["LogP"].std())
    logp_statistics.append(df["LogP"].skew())
    logp_statistics.append(df["LogP"].kurt())
    logp_statistics.append(diversity)

    sa_statistics.append(df["SA"].mean())
    sa_statistics.append(df["SA"].median())
    sa_statistics.append(df["SA"].quantile(q=0.25))
    sa_statistics.append(df["SA"].quantile(q=0.75))
    sa_statistics.append(df["SA"].std())
    sa_statistics.append(df["SA"].skew())
    sa_statistics.append(df["SA"].kurt())
    sa_statistics.append(diversity)

    tpsa_statistics.append(df["TPSA"].mean())
    tpsa_statistics.append(df["TPSA"].median())
    tpsa_statistics.append(df["TPSA"].quantile(q=0.25))
    tpsa_statistics.append(df["TPSA"].quantile(q=0.75))
    tpsa_statistics.append(df["TPSA"].std())
    tpsa_statistics.append(df["TPSA"].skew())
    tpsa_statistics.append(df["TPSA"].kurt())
    tpsa_statistics.append(diversity)

    pd.DataFrame({"logp": logp_statistics, "tpsa": tpsa_statistics, "sa": sa_statistics}, index=indices).\
        to_csv(out_path, float_format="%.3f")
    return out_path


def _iter_seed_files(objective, prop, prefix):
    folder = os.path.join(PSO_RESULTS_DIR, objective, prop)
    pattern = os.path.join(folder, f"{prefix}-Seed*.npz")
    rgx = re.compile(r"-Seed(\d+)\.npz$")
    seeds = []
    for fp in glob.glob(pattern):
        m = rgx.search(fp)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def evaluate_objective(objective, tasks, seed_start=0, n_seeds=100, strict=False, force=False):
    summary = []
    for prop, target in tasks:
        expected = list(range(seed_start, seed_start + n_seeds))
        metrics = psoValidity(objective=objective, prop=prop, prefix=target,
                              seed_list=expected, strict=strict)
        psoGeneratedDistribution(objective=objective, prop=prop, prefix=target, force=force)
        psoDescription(objective=objective, prop=prop, prefix=target, force=force)
        summary.append(metrics)

    summary_df = pd.DataFrame(summary)
    out = os.path.join(PSO_RESULTS_DIR, objective, "evaluation_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    summary_df.to_csv(out, index=False)
    return out


def combine_round_results(base_objective, objectives, tasks):
    rows = []
    for prop, target in tasks:
        validity_frames = []
        property_frames = []
        for obj in objectives:
            v = os.path.join(PSO_RESULTS_DIR, obj, prop, f"{target} Validity.csv")
            p = os.path.join(PSO_RESULTS_DIR, obj, prop, f"{target} Property.csv")
            if os.path.exists(v):
                validity_frames.append(pd.read_csv(v))
            if os.path.exists(p):
                property_frames.append(pd.read_csv(p))

        if not validity_frames:
            continue

        out_dir = os.path.join(PSO_RESULTS_DIR, base_objective, prop)
        os.makedirs(out_dir, exist_ok=True)

        combined_validity = pd.concat(validity_frames, ignore_index=True)
        combined_validity.to_csv(os.path.join(out_dir, f"{target} Validity.csv"), index=False)

        if property_frames:
            combined_property = pd.concat(property_frames, ignore_index=True)
            combined_property.to_csv(os.path.join(out_dir, f"{target} Property.csv"), index=False)
            psoDescription(objective=base_objective, prop=prop, prefix=target, force=True)

        validity_percent = combined_validity["Validity"].mean() * 100
        diversity_percent = combined_validity["SMILES"].drop_duplicates().shape[0] * 100 / len(combined_validity)
        rows.append({
            "objective": base_objective,
            "property": prop,
            "target": target,
            "decoded_smiles": len(combined_validity),
            "validity_percent": validity_percent,
            "diversity_percent": diversity_percent,
            "sources": ";".join(objectives),
        })

    out = os.path.join(PSO_RESULTS_DIR, base_objective, "evaluation_summary.csv")
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run_chunked_evaluation(rounds=5, chunk_size=100, include_uni=True, include_multi=True, force=False):
    created = []
    if include_uni:
        uni_objectives = [f"UNI-OBJECTIVE-R{i}" for i in range(1, rounds + 1)]
        for i, objective in enumerate(uni_objectives, start=1):
            created.append(evaluate_objective(objective=objective,
                                             tasks=UNI_TASKS,
                                             seed_start=(i - 1) * chunk_size,
                                             n_seeds=chunk_size,
                                             strict=False,
                                             force=force))
        created.append(combine_round_results(base_objective="UNI-OBJECTIVE",
                                            objectives=uni_objectives,
                                            tasks=UNI_TASKS))

    if include_multi:
        tri_objectives = [f"MULTI-OBJECTIVE-R{i}" for i in range(1, rounds + 1)]
        for i, objective in enumerate(tri_objectives, start=1):
            created.append(evaluate_objective(objective=objective,
                                             tasks=TRI_TASKS,
                                             seed_start=(i - 1) * chunk_size,
                                             n_seeds=chunk_size,
                                             strict=False,
                                             force=force))
        created.append(combine_round_results(base_objective="MULTI-OBJECTIVE",
                                            objectives=tri_objectives,
                                            tasks=TRI_TASKS))

    return created


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PSO outputs and generate per-round + combined reports.")
    parser.add_argument("--chunked-eval", action="store_true", help="Evaluate R1..Rn chunk folders and combine them.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of chunk rounds (R1..Rn).")
    parser.add_argument("--chunk-size", type=int, default=100, help="Seeds per round.")
    parser.add_argument("--skip-uni", action="store_true", help="Skip UNI-OBJECTIVE-R* evaluation.")
    parser.add_argument("--skip-multi", action="store_true", help="Skip MULTI-OBJECTIVE-R* evaluation.")
    parser.add_argument("--force", action="store_true", help="Force rebuilding property/description CSVs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.chunked_eval:
        outputs = run_chunked_evaluation(rounds=args.rounds,
                                         chunk_size=args.chunk_size,
                                         include_uni=not args.skip_uni,
                                         include_multi=not args.skip_multi,
                                         force=args.force)
        print("Chunked evaluation finished.")
        for p in outputs:
            print(f"- {p}")
        raise SystemExit(0)

    if not os.path.exists(os.path.join(MOSES_DS_DIR, "testPrediction.csv")):
        makePredictions()
    if not os.path.exists(os.path.join(RESULTS_MODELS_QSPR_DIR, "performance.txt")):
        statistics()
    # VAE_Validity()
    # for name in ["SeqQSPR(LOGP)_Refined_Embedding", "SeqQSPR(SA)_Refined_Embedding", "SeqQSPR(TPSA)_Refined_Embedding",
    #              "VAE_Refined_Embedding", "Word2vecEmbedding"]:
        # token_similarity(name)
    # -----------------------------------------------------------
    for o,p,t in [["MULTI-OBJECTIVE", "LOGP-TPSA", "1.0-20.0"], ["MULTI-OBJECTIVE", "LOGP-TPSA", "2.0-40.0"],
                  ["MULTI-OBJECTIVE", "LOGP-TPSA", "3.0-60.0"], ["MULTI-OBJECTIVE", "LOGP-TPSA", "4.0-80.0"],
                  ["MULTI-OBJECTIVE", "LOGP", "1.0"], ["MULTI-OBJECTIVE", "LOGP", "2.0"],
                  ["MULTI-OBJECTIVE", "LOGP", "3.0"], ["MULTI-OBJECTIVE", "LOGP", "4.0"],
                  ["MULTI-OBJECTIVE", "TPSA", "20.0"], ["MULTI-OBJECTIVE", "TPSA", "40.0"],
                  ["MULTI-OBJECTIVE", "TPSA", "60.0"], ["MULTI-OBJECTIVE", "TPSA", "80.0"],
                  ["UNI-OBJECTIVE", "LOGP", "1.0"], ["UNI-OBJECTIVE", "LOGP", "2.0"],
                  ["UNI-OBJECTIVE", "LOGP", "3.0"], ["UNI-OBJECTIVE", "LOGP", "4.0"],
                  ["UNI-OBJECTIVE", "TPSA", "20.0"], ["UNI-OBJECTIVE", "TPSA", "40.0"],
                  ["UNI-OBJECTIVE", "TPSA", "60.0"], ["UNI-OBJECTIVE", "TPSA", "80.0"]]:
        if not os.path.exists(os.path.join(PSO_RESULTS_DIR, o, p, f"{t} Validity.csv")):
            psoValidity(o, p, t)
        psoGeneratedDistribution(o, p, t)
        if not os.path.exists(os.path.join(PSO_RESULTS_DIR, o, p, f"{t} description@10K.csv")):
            psoDescription(o, p, t)
    # -----------------------------------------------------------
    # data_description()
    # -----------------------------------------------------------