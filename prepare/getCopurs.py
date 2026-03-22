from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import os
import re
from requests import request


def make_chembl_csv():
    suppl = Chem.SDMolSupplier('chembl_31/chembl_31.sdf')
    with open("chembl_31/chembl_31.csv", "wb") as csvfile:
        print("Start writing SMILES notations...")
        regex = r"[abdefghijkmqtuvwxyzADEGIJKLMQRTVWXYZ]|\[\d{1,3}\D{1,3}\]|[^B]r"
        pattern = re.compile(regex)
        for i in tqdm(suppl):
            smi = Chem.MolToSmiles(i)
            if i.GetRingInfo().NumRings() < 5 and Descriptors.MolWt(i) < 250 and not pattern.search(smi):
                csvfile.write(smi.encode(encoding="utf-8"))
                csvfile.write("\n".encode(encoding="utf-8"))


if not os.path.exists("chembl_31/chembl_31.csv"):
    make_chembl_csv()


def make_zinc_csv():
    urls = open("zinc/zinc_downloader.uri").readlines()
    paths = ["zinc/"+i[-9:] for i in urls]
    regex = r"[abdefghijkmqtuvwxyzADEGIJKLMQRTVWXYZ]|\[\d{1,3}\D{1,3}\]|[^B]r"
    pattern = re.compile(regex)
    print("Start writing SMILES notations...")
    with open("zinc/zinc.csv", "wb") as csvfile:
        for idx, path in enumerate(paths):
            print(f"Getting SMILES from {path.strip().split('/')[1]}")
            print(f">>>>>{idx}/{len(urls)}<<<<<")
            with open(path.strip(), encoding="utf-8") as file:
                for idx, line in tqdm(enumerate(file)):
                    if idx > 0:
                        try:
                            i = Chem.MolFromSmiles(line.split("\t")[0])
                            smi = Chem.MolToSmiles(i)
                        except:
                            continue
                        if i.GetRingInfo().NumRings() < 5 and Descriptors.MolWt(i) < 250 and not pattern.search(smi):
                            csvfile.write(smi.encode(encoding="utf-8"))
                            csvfile.write("\n".encode(encoding="utf-8"))


if not os.path.exists("zinc/zinc.csv"):
    make_zinc_csv()


if not os.path.exists("jointCorpus/SMILESCorpus.csv"):
    with open("jointCorpus/SMILESCorpus.csv", "wb") as smicsv:
        with open("chembl_31/chembl_31.csv", encoding="utf-8") as chembl:
            for line in tqdm(chembl):
                smicsv.write(line.encode("utf-8"))
        with open("zinc/zinc.csv", encoding="utf-8") as zinc:
            for line in tqdm(zinc):
                smicsv.write(line.encode("utf-8"))