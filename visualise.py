import os
import random
import numpy as np
from os import path
import pandas as pd
from PIL import Image
import seaborn as sns
from vae import CNNVAE
from rdkit import Chem
import tensorflow as tf
from dpcnn import SeqQSPR
from rdkit.Chem import Draw
from prepare import Tokenize
from functools import partial
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from wordcloud import WordCloud, ImageColorGenerator
from config import (
    MOSES_DS_DIR,
    MOL_PROPERTIES_DIR,
    TOKEN_SET_DIR,
    EMBEDDING_WEIGHTS_DIR,
    RESULTS_FEATURE_DIR,
    RESULTS_EXAMPLE_MOLS_DIR,
    RESULTS_LATENTS_DIR,
    PSO_RESULTS_DIR,
    set_working_directory,
    ensure_dir,
)

set_working_directory()
ensure_dir(RESULTS_FEATURE_DIR)
ensure_dir(RESULTS_EXAMPLE_MOLS_DIR)
ensure_dir(RESULTS_LATENTS_DIR)

colors = ["#82AFF9", "#118DF0", "#9881F5", "#2D248A", "#08FFC8",
          "#28CC75", "#F97D81", "#FF420E", "#FED95C", "#F29C2B"]
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


class VAEDatasetIterater(object):
    def __init__(self, dataname="mosesDs", label="test", max_len=60, dataCapacity=1e4):
        Path = f"{dataname}/{label}.csv"
        self.maxlen = max_len
        dataset = []
        self.tokenizer = Tokenize()
        with open(Path, encoding="utf-8") as file:
            for idx, line in tqdm(enumerate(file)):
                if 1 <= idx <= dataCapacity:
                    print(f"\r\033[1;37;40m{line.strip()}\033[0m", end="")
                    smi = self.tokenizer.tokenize(line.strip())
                    if len(smi) <= self.maxlen:
                        dataset.append(smi)
        self.dataset = []
        for line in tqdm(dataset):
            try:
                line = [int(self.tokenizer.t2i[i]) for i in line]
                seqlen = len(line)
                line += (self.maxlen - seqlen) * [0]
                self.dataset.append(line)
            except KeyError:
                continue

    def _to_tensor(self, x):
        x = tf.constant(x)
        return x

    def __getitem__(self, idx):
        return self._to_tensor([self.dataset[idx]])

    def getstr(self, idx):
        return "".join(self.tokenizer.detokenize(self.dataset[idx]))

    def __len__(self):
        return len(self.dataset)


class QSPRDatasetIterater(object):
    def __init__(self, dataname="LOGP", label="test", max_len=60, dataCapacity=1e4):
        Path = os.path.join(MOL_PROPERTIES_DIR, dataname, f"{label}.csv")
        self.maxlen = max_len
        dataset = []
        self.tokenizer = Tokenize()
        with open(Path, encoding="utf-8") as file:
            for idx, line in tqdm(enumerate(file)):
                if idx <= dataCapacity-1:
                    smi, y = line.strip().split(",")
                    print(f"\r\033[1;37;40m{smi}\033[0m {dataname}:{y}", end="")
                    smi = self.tokenizer.tokenize(smi)
                    if len(smi) <= self.maxlen:
                        dataset.append((smi, y))
        self.dataset = []
        for smi, y in tqdm(dataset):
            try:
                smi = [int(self.tokenizer.t2i[i]) for i in smi]
                seqlen = len(smi)
                smi += (self.maxlen - seqlen) * [0]
                self.dataset.append([smi, eval(y)])
            except KeyError:
                continue

    def _to_tensor(self, x):
        x = tf.constant(x)
        return x

    def __getitem__(self, idx):
        return (self._to_tensor([self.dataset[idx][0]]), self._to_tensor(self.dataset[idx][1]))

    def getstr(self, idx):
        return "".join(self.tokenizer.detokenize(self.dataset[idx][0]))

    def __len__(self):
        return len(self.dataset)


dataCapacity = 1e8
moses_train = VAEDatasetIterater(label="train", dataCapacity=dataCapacity)
moses_test = VAEDatasetIterater(label="test", dataCapacity=dataCapacity)
sa_train = QSPRDatasetIterater(dataname="SA", label="train", dataCapacity=dataCapacity)
sa_test = QSPRDatasetIterater(dataname="SA", label="test", dataCapacity=dataCapacity)
logp_train = QSPRDatasetIterater(dataname="LOGP", label="train", dataCapacity=dataCapacity)
logp_test = QSPRDatasetIterater(dataname="LOGP", label="test", dataCapacity=dataCapacity)
tpsa_train = QSPRDatasetIterater(dataname="TPSA", label="train", dataCapacity=dataCapacity)
tpsa_test = QSPRDatasetIterater(dataname="TPSA", label="test", dataCapacity=dataCapacity)
cnnvae = CNNVAE()
cnnvae.load()
encoder = cnnvae.encoder
decoder = cnnvae.decoder


def example_molecules():
    count = 0
    tokenizer = Tokenize()
    tokenized = []
    with open(os.path.join(MOSES_DS_DIR, "test.csv"), encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if count == 100:
                break
            try:
                mol = Chem.MolFromSmiles(line.strip())
                tokenized.append(line.strip())
                out_img = os.path.join(RESULTS_EXAMPLE_MOLS_DIR, f"{idx-1}.bmp")
                if not os.path.exists(out_img):
                    img = Draw.MolToImage(mol)
                    img.save(out_img, quality=10000)
                count += 1
            except:
                pass
    with open(os.path.join(RESULTS_EXAMPLE_MOLS_DIR, "tokenized-Mols.csv"), "wb") as csv_writer:
        for mol in tokenized:
            mol = tokenizer.tokenize(mol)
            mol = ", ".join(mol)
            csv_writer.write(mol.encode("utf-8"))
            csv_writer.write("\n".encode("utf-8"))


def SMILES_length_histogram():
    train_length = []
    test_length = []
    for i in tqdm(moses_train.dataset):
        train_length.append(len(i)-i.count(0))
    for i in tqdm(moses_test.dataset):
        test_length.append(len(i)-i.count(0))

    left = min(train_length+test_length)
    right = max(train_length+test_length)

    fontsize = 20
    fig, axes = plt.subplots(2, 1)
    rc_parameters = {"figure.figsize": [10, 10], "font.sans-serif": "Times New Roman"}
    sns.set_theme(style='white', font='Times New Roman', rc=rc_parameters)
    # bins = 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'
    sns.histplot(train_length, bins=20, ax=axes[0], color="#6A5ACD", line_kws={"linewidth":3})
    sns.histplot(test_length, bins=20, ax=axes[1], color="#FF6347", line_kws={"linewidth":3})
    axes[0].legend(["Train"], fontsize=fontsize)
    axes[1].legend(["Test"], fontsize=fontsize)

    axes[0].xaxis.set_major_locator(ticker.FixedLocator(axes[0].get_xticks().tolist()))
    axes[0].set_xticklabels(["%d"%i for i in axes[0].get_xticks().tolist()], fontsize=fontsize)
    axes[0].set_xlim(left, right)
    axes[0].set_yticks([0, 6e4, 1.2e5, 1.8e5, 2.4e5, 3e5])
    axes[0].set_yticklabels(["0", "60K", "120K", "180K", "240K", "300K"], fontsize=fontsize)
    axes[0].set_xlabel("Length", fontsize=fontsize)
    axes[0].set_ylabel("Counts", fontsize=fontsize)

    axes[1].xaxis.set_major_locator(ticker.FixedLocator(axes[1].get_xticks().tolist()))
    axes[1].set_xticklabels(["%d"%i for i in axes[1].get_xticks().tolist()], fontsize=fontsize)
    axes[1].set_xlim(left, right)
    axes[1].set_yticks([0, 6e3, 1.2e4, 1.8e4, 2.4e4, 3e4])
    axes[1].set_yticklabels(["0", "6K", "12K", "18K", "24K", "30K"], fontsize=fontsize)
    axes[1].set_xlabel("Length", fontsize=fontsize)
    axes[1].set_ylabel("Counts", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FEATURE_DIR, "Length Distribution.png"), dpi=1000)


def data_histogram(data_type):
    train_data = []
    test_data = []

    if data_type.upper() == "LOGP":
        train, test = logp_train, logp_test
    elif data_type.upper() == "TPSA":
        train, test = tpsa_train, tpsa_test
    else:
        train, test = sa_train, sa_test

    for i in tqdm(train.dataset):
        train_data.append(i[1])
    for i in tqdm(test.dataset):
        test_data.append(i[1])

    left = min(train_data+test_data)
    right = max(train_data+test_data)
    fontsize = 28
    fig, axes = plt.subplots(2, 1)
    rc_parameters = {"figure.figsize": [10, 10], "font.sans-serif": "Times New Roman"}
    sns.set_theme(style='white', font='Times New Roman', rc=rc_parameters)
    # bins = 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'
    sns.histplot(train_data, bins=20, ax=axes[0], color="#6A5ACD", line_kws={"linewidth":3})
    sns.histplot(test_data, bins=20, ax=axes[1], color="#FF6347", line_kws={"linewidth":3})
    axes[0].legend(["Train"], fontsize=fontsize)
    axes[1].legend(["Test"], fontsize=fontsize)

    axes[0].xaxis.set_major_locator(ticker.FixedLocator(axes[0].get_xticks().tolist()))
    if data_type.upper() == "TPSA":
        axes[0].set_xticklabels(["%d"%i for i in axes[0].get_xticks().tolist()], fontsize=fontsize)
    else:
        axes[0].set_xticklabels(["%.1f"%i for i in axes[0].get_xticks().tolist()], fontsize=fontsize)
    axes[0].set_xlim(left, right)
    axes[0].set_yticks([0, 1e5, 2e5, 3e5, 4e5, 5e5])
    axes[0].set_yticklabels(["0", "100K", "200K", "300K", "400K", "500K"], fontsize=fontsize)
    axes[0].set_xlabel(f"{data_type}", fontsize=fontsize)
    axes[0].set_ylabel("Counts", fontsize=fontsize)

    axes[1].xaxis.set_major_locator(ticker.FixedLocator(axes[1].get_xticks().tolist()))
    if data_type.upper() == "TPSA":
        axes[1].set_xticklabels(["%d"%i for i in axes[1].get_xticks().tolist()], fontsize=fontsize)
    else:
        axes[1].set_xticklabels(["%.1f"%i for i in axes[1].get_xticks().tolist()], fontsize=fontsize)
    axes[1].set_xlim(left, right)
    axes[1].set_yticks([0, 1e4, 2e4, 3e4, 4e4, 5e4])
    axes[1].set_yticklabels(["0", "10K", "20K", "30K", "40K", "50K"], fontsize=fontsize)
    axes[1].set_xlabel(f"{data_type}", fontsize=fontsize)
    axes[1].set_ylabel("Counts", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FEATURE_DIR, f"{data_type} Distribution.png"), dpi=1000)


def token_frequency_and_cloud():
    frequency_path = os.path.join(TOKEN_SET_DIR, "cleanedUnigramFreq.txt")
    token_frequency = {"INDEX": [], "TOKEN": [], "FREQUENCY": []}
    with open(frequency_path, encoding="utf-8") as csv_reader:
        for idx, line in enumerate(tqdm(csv_reader)):
            token_frequency["INDEX"] += [idx]
            token_frequency["TOKEN"] += [line.strip().split(":")[0]]
            frequency = eval(line.strip().split(":")[1])
            token_frequency["FREQUENCY"] += [frequency]

    df = pd.DataFrame(token_frequency)
    df["FREQUENCY"] = df["FREQUENCY"].map(lambda x: np.log10(x))
    df.set_index("INDEX", inplace=True)
    t2f = {df.loc[i]["TOKEN"]: df.loc[i]["FREQUENCY"] for i in range(1500)}

    wc = WordCloud(background_color="white", max_words=1500, width=1600, height=1200, colormap="gist_rainbow")
    wc.generate_from_frequencies(t2f)
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig(os.path.join(RESULTS_FEATURE_DIR, "TokenCloud.png"), dpi=3000)


def embedding_visualization(type):
    # The former 100 tokens.
    # 10 embedding matrices of Word2vec + VAE + 2 * (QED + LOGP + TPSA + SA).
    # 10 similarity matrices of Word2vec + VAE + 2 * (QED + LOGP + TPSA + SA).
    query_tokens = pd.read_csv(os.path.join(TOKEN_SET_DIR, "cleanedUnigramFreq.txt"), delimiter=":", header=None)
    query_tokens.columns = ["Tokens", "Frequency"]
    query_tokens = query_tokens["Tokens"].values.tolist()[:100]
    tokenizer = Tokenize()
    if type == "Word2vec":
        path = os.path.join(EMBEDDING_WEIGHTS_DIR, "Word2vecEmbedding.csv")
        embedding = pd.read_csv(path, index_col="Token")
        embedding = pd.DataFrame(np.concatenate((np.zeros((1, embedding.shape[1])), embedding.values), axis=0))
    else:
        path = os.path.join(EMBEDDING_WEIGHTS_DIR, f"{type}_Refined_Embedding.csv")
        try:
            embedding = pd.read_csv(path)
        except FileNotFoundError:
            print("This type of embedding doesn't exists, now loading the default glove embedding matrix.")
            return
    type = type.upper()
    embedding.index = ["<PAD>"] + list(tokenizer.t2i.keys())

    if not os.path.exists(os.path.join(RESULTS_FEATURE_DIR, f"{type}-Embedding.png")):
        # The heat map of the embedding matrix.
        matrix = np.array([embedding.loc[i] for i in query_tokens])
        fig, ax = plt.subplots(1, 1, figsize=(15, 20))
        rc_parameters = {"font.sans-serif": "Times New Roman"}
        sns.set_theme(font='Times New Roman', rc=rc_parameters)
        ax = sns.heatmap(matrix, ax=ax, vmax=1, vmin=-1, cbar=False,
                         cmap=sns.diverging_palette(h_neg=250, h_pos=10, s=100, l=50, sep=3, n=100, center="light"))
        cb = ax.figure.colorbar(ax.collections[0], shrink=0.4)
        cb.ax.tick_params(labelsize=14)
        ax.set_yticks(np.array(range(100)) + 0.5)
        ax.set_yticklabels(query_tokens, rotation=0, fontsize=8)
        ax.tick_params(bottom=False)
        ax.set_xticks(np.array(range(100)) + 0.5)
        ax.set_xticklabels([""]*len(ax.get_xticklabels()))
        ax.set_xlabel("Token embedding vector", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FEATURE_DIR, f"{type}-Embedding.png"), dpi=1000)
        plt.cla()
        plt.clf()

    if not os.path.exists(os.path.join(RESULTS_FEATURE_DIR, f"{type}-Embedding-Similarity.png")):
        # The heat map of the token similarity.
        i_j_similarity = []
        for token_i in query_tokens:
            _ = []
            for token_j in query_tokens:
                _.append(1- cosine(embedding.loc[token_i], embedding.loc[token_j]))
            i_j_similarity.append(_)
        matrix = np.array(i_j_similarity)
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        rc_parameters = {"font.sans-serif": "Times New Roman"}
        sns.set_theme(font='Times New Roman', rc=rc_parameters)
        ax = sns.heatmap(matrix, ax=ax, vmax=1, vmin=0, cbar=False, square=True,
                         annot=True, annot_kws={"fontsize": 5}, fmt=".2f",  cmap=sns.color_palette("Reds", 100))
        cb = ax.figure.colorbar(ax.collections[0], shrink=0.2)
        cb.ax.tick_params(labelsize=14)
        ax.set_yticks(np.array(range(100)) + 0.5)
        ax.set_yticklabels(query_tokens, rotation=0, fontsize=8)
        ax.tick_params(axis="x", bottom=False, top=True)
        ax.set_xticks(np.array(range(100)) + 0.5)
        ax.set_xticklabels(query_tokens, rotation=90, fontsize=8)
        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FEATURE_DIR, f"{type}-Embedding-Similarity.png"), dpi=1000)
        plt.cla()
        plt.clf()


def vae_latents_visualization():
    color = "#118DF0"
    # visualize the latents distribution of the test set molecules.
    from sklearn.manifold import TSNE
    TSNE = TSNE(n_components=2)
    latents = []
    for i in tqdm(moses_test.dataset):
        latent, *_ = encoder(moses_test._to_tensor([i]))
        latent = latent.numpy().squeeze()
        latents.append(latent)
    latents = np.array(latents)
    data = []
    for i in trange(0, len(latents), 1000):
        data.append(TSNE.fit_transform(latents[i:i+1000]))

    data = pd.DataFrame(np.concatenate(data), columns=["Decomposed dimension 1", "Decomposed dimension 2"])
    data.to_csv(os.path.join(RESULTS_LATENTS_DIR, "VAE Latents.csv"))
    rc_parameters = {"font.sans-serif": "Times New Roman"}
    sns.set_theme(font='Times New Roman', rc=rc_parameters, font_scale=2, style="ticks")
    g = sns.JointGrid(x=data["Decomposed dimension 1"], y=data["Decomposed dimension 2"], height=15,
                      xlim=(-25, 25), ylim=(-25, 25), ratio=4)
    g.plot_joint(sns.scatterplot, s=200, alpha=.1, color=color, marker="*")
    g.plot_marginals(sns.kdeplot, fill=True, color=color, linewidth=5)
    g.refline(x=0, y=0, color=color, linewidth=5, linestyle="dashed")
    g.ax_marg_y.spines["left"].set_linewidth(3)
    g.ax_marg_x.spines["bottom"].set_linewidth(3)
    g.ax_joint.spines["left"].set_linewidth(3)
    g.ax_joint.spines["bottom"].set_linewidth(3)
    g.ax_joint.xaxis.set_ticks(list(range(-25, 26, 5)))
    g.ax_joint.yaxis.set_ticks(list(range(-25, 26, 5)))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_LATENTS_DIR, "VAE Latents Visualization.png"), dpi=600)
    plt.clf()


if not os.path.exists(os.path.join(RESULTS_EXAMPLE_MOLS_DIR, "tokenized-Mols.csv")):
    example_molecules()
if not os.path.exists(os.path.join(RESULTS_FEATURE_DIR, "Length Distribution.png")):
    SMILES_length_histogram()
for dt in ["LogP", "TPSA", "SA"]:
    if not os.path.exists(os.path.join(RESULTS_FEATURE_DIR, f"{dt} Distribution.png")):
        data_histogram(dt)
if not os.path.exists(os.path.join(RESULTS_FEATURE_DIR, "TokenCloud.png")):
    token_frequency_and_cloud()
for type in ["Word2vec", "VAE", "SeqQSPR(LOGP)", "SeqQSPR(TPSA)", "SeqQSPR(SA)"]:
    embedding_visualization(type)


def drawBestTenMol(objective="MULTI-OBJECTIVE", prop="LogP", prefix="2.0"):
    df = pd.read_csv(os.path.join(PSO_RESULTS_DIR, objective, prop.upper(), f"{prefix} Property.csv"))
    df.drop_duplicates(inplace=True)
    prop_ = prop
    prop = prop.split("-")
    target = prefix.split("-")
    if objective.upper() == "MULTI-OBJECTIVE" and len(prop) == 1:
        df["fitness"] = df["SA"] * np.abs((df[prop[0]] / float(target[0]) - 1))
        df.sort_values(by="fitness", inplace=True)
        df.index = range(len(df))
        for i in range(50):
            mol = Chem.MolFromSmiles(df.loc[i]['SMILES'])
            img = Draw.MolToImage(mol)
            mols_dir = ensure_dir(os.path.join(PSO_RESULTS_DIR, objective, prop_.upper(), "Mols"))
            img.save(os.path.join(mols_dir, f"SA{round(df.loc[i]['SA'], 2)}-{prop[0]}{round(df.loc[i][prop[0]], 2)}.bmp"), quality=10000)
    elif objective.upper() == "MULTI-OBJECTIVE" and len(prop) == 2:
        df["fitness"] = df["SA"] * np.abs(df[prop[0]] / float(target[0]) + df[prop[1]] / float(target[1]) - 2)
        df.sort_values(by="fitness", inplace=True)
        df.index = range(len(df))
        for i in range(50):
            mol = Chem.MolFromSmiles(df.loc[i]['SMILES'])
            img = Draw.MolToImage(mol)
            mols_dir = ensure_dir(os.path.join(PSO_RESULTS_DIR, objective, prop_.upper(), "Mols"))
            img.save(os.path.join(mols_dir, f"SA{round(df.loc[i]['SA'],2)}-{prop[0]}{round(df.loc[i][prop[0]], 2)}-{prop[1]}{round(df.loc[i][prop[1]], 2)}.bmp"), quality=10000)
    else:
        df["fitness"] = np.abs((df[prop[0]] / float(target[0]) - 1))
        df.sort_values(by="fitness", inplace=True)
        df.index = range(len(df))
        for i in range(50):
            mol = Chem.MolFromSmiles(df.loc[i]['SMILES'])
            img = Draw.MolToImage(mol)
            mols_dir = ensure_dir(os.path.join(PSO_RESULTS_DIR, objective, prop_.upper(), "Mols"))
            img.save(os.path.join(mols_dir, f"{prop[0]}{round(df.loc[i][prop[0]], 2)}.bmp"), quality=10000)


if __name__ == "__main__":
    # vae_latents_visualization()
    for o,p,t in [["MULTI-OBJECTIVE", "LogP-TPSA", "1.0-20.0"], ["MULTI-OBJECTIVE", "LogP-TPSA", "2.0-40.0"],
                  ["MULTI-OBJECTIVE", "LogP-TPSA", "3.0-60.0"], ["MULTI-OBJECTIVE", "LogP-TPSA", "4.0-80.0"],
                  ["MULTI-OBJECTIVE", "LogP", "1.0"], ["MULTI-OBJECTIVE", "LogP", "2.0"],
                  ["MULTI-OBJECTIVE", "LogP", "3.0"], ["MULTI-OBJECTIVE", "LogP", "4.0"],
                  ["MULTI-OBJECTIVE", "TPSA", "20.0"], ["MULTI-OBJECTIVE", "TPSA", "40.0"],
                  ["MULTI-OBJECTIVE", "TPSA", "60.0"], ["MULTI-OBJECTIVE", "TPSA", "80.0"],
                  ["UNI-OBJECTIVE", "LogP", "1.0"], ["UNI-OBJECTIVE", "LogP", "2.0"],
                  ["UNI-OBJECTIVE", "LogP", "3.0"], ["UNI-OBJECTIVE", "LogP", "4.0"],
                  ["UNI-OBJECTIVE", "TPSA", "20.0"], ["UNI-OBJECTIVE", "TPSA", "40.0"],
                  ["UNI-OBJECTIVE", "TPSA", "60.0"], ["UNI-OBJECTIVE", "TPSA", "80.0"]]:
        drawBestTenMol(o, p, t)