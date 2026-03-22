import os
import re
import seaborn
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from prepare.smiTokenizer import Tokenize
from config import TOKEN_SET_DIR, JOINT_CORPUS_DIR, RESULTS_TOKENS_DIR, set_working_directory, ensure_dir


set_working_directory()
ensure_dir(RESULTS_TOKENS_DIR)


def cleanTokenization():
    with open(os.path.join(TOKEN_SET_DIR, "rawUnigram.vocab"), "r") as file:
        with open(os.path.join(TOKEN_SET_DIR, "cleanedUnigram.txt"), "wb") as file_:
            for idx, line in enumerate(tqdm(file)):
                if idx >= 3:
                    tkn = line.split("\t")[0]
                    if Chem.MolFromSmiles(tkn):
                        print(f"\r Current token is \033[1;37;40m{tkn}\033[0m and it is valid :)", end="")
                        tkn+="\n"
                        file_.write(tkn.encode("utf-8"))
                    else:
                        print(f"\r Current token is \033[1;37;40m{tkn}\033[0m "
                              f"and it is invalid. Hence it is filtered :(", end="")


if not os.path.exists(os.path.join(TOKEN_SET_DIR, "cleanedUnigram.txt")):
    cleanTokenization()


tokenizer = Tokenize(dictPath=os.path.join(TOKEN_SET_DIR, "cleanedUnigram.txt"),
                     tokenPath=None, forEmbedding=False)
if not os.path.exists(os.path.join(JOINT_CORPUS_DIR, "rawUnigramCorpus.csv")):
    with open(os.path.join(JOINT_CORPUS_DIR, "rawUnigramCorpus.csv"), "wb") as file1:
        with open(os.path.join(JOINT_CORPUS_DIR, "SMILESCorpus.csv"), encoding="utf-8") as file2:
            for line in tqdm(file2):
                file1.write(" ".join(tokenizer.tokenize(line)).encode("utf-8"))


if not os.path.exists(os.path.join(TOKEN_SET_DIR, "cleanedUnigramFreq.txt")):
    tokenFreq = {}
    with open(os.path.join(JOINT_CORPUS_DIR, "rawUnigramCorpus.csv"), encoding="utf-8") as file:
        for line in tqdm(file):
            line = line.strip().split()
            for t in line:
                tokenFreq[t] = tokenFreq.get(t,0) + 1
    tokenFreq = sorted(list(tokenFreq.items()), key=lambda x: x[1], reverse=True)
    tokens = [i.strip() for i in open(os.path.join(TOKEN_SET_DIR, "cleanedUnigram.txt"),
                                      encoding="utf-8").readlines()]
    with open(os.path.join(TOKEN_SET_DIR, "cleanedUnigramFreq.txt"), "wb") as file:
        for k,v in tokenFreq:
            if k in tokens:
                file.write(f"{k}:{v}\n".encode("utf-8"))


if not os.path.exists(os.path.join(TOKEN_SET_DIR, "Unigram.csv")):
    print("Plotting molecular tokens graphs.")
    with open(os.path.join(TOKEN_SET_DIR, "Unigram.csv"), "wb") as file:
        vocabs_ = [(i.strip().split(":")) for i in open(os.path.join(TOKEN_SET_DIR, "cleanedUnigramFreq.txt"),
                                                        encoding="utf-8").readlines()]
        vocabs_ = sorted(vocabs_, key=lambda x: eval(x[1]), reverse=True)
        for idx, i in enumerate(vocabs_[:1500]):
            img = Draw.MolToImage(Chem.MolFromSmiles(i[0]))
            img.save(os.path.join(RESULTS_TOKENS_DIR, f"{idx}.bmp"))
            i = i[0] + "\n"
            file.write(i.encode("utf-8"))


tokenizer = Tokenize(dictPath=os.path.join(TOKEN_SET_DIR, "Unigram.csv"),
                     tokenPath=None, forEmbedding=False)
if not os.path.exists(os.path.join(JOINT_CORPUS_DIR, "UnigramCorpus.csv")):
    with open(os.path.join(JOINT_CORPUS_DIR, "UnigramCorpus.csv"), "wb") as file1:
        with open(os.path.join(JOINT_CORPUS_DIR, "SMILESCorpus.csv"), encoding="utf-8") as file2:
            for line in tqdm(file2):
                line = tokenizer.tokenize(line)
                file1.write(" ".join(line).encode("utf-8"))
if not os.path.exists(os.path.join(TOKEN_SET_DIR, "tokenList.csv")):
    print("Making tokenList.csv")
    tokenFreq = {}
    with open(os.path.join(JOINT_CORPUS_DIR, "UnigramCorpus.csv"), encoding="utf-8") as file1:
        for line in tqdm(file1):
            line = line.strip().split()
            if "r" not in line and "6" not in line and "5" not in line and "p" not in line and "b" not in line:
                for t in line:
                    tokenFreq[t] = tokenFreq.get(t, 0) + 1
    tokenFreq = sorted(list(tokenFreq.items()), key=lambda x: x[1], reverse=True)
    with open(os.path.join(TOKEN_SET_DIR, "tokenList.csv"), "wb") as file:
        for k,v in tokenFreq:
            file.write(f"{k}:{v}\n".encode("utf-8"))
