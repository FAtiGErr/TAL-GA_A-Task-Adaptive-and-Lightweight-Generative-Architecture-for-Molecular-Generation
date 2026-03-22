import os
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from prepare import Tokenize
from rdkit.ML.Descriptors import MoleculeDescriptors
from prepare.synava import SaScore
from config import TOKEN_SET_DIR, MOSES_DS_DIR, MOSES_DIR, MOL_PROPERTIES_DIR, set_working_directory


set_working_directory()


def _moses_dir():
    # Keep backward compatibility with projects that still use `moses/`.
    return MOSES_DIR if os.path.exists(MOSES_DIR) else MOSES_DS_DIR


tokenizer = Tokenize(
    dictPath=os.path.join(TOKEN_SET_DIR, "Unigram.csv"),
    tokenPath=os.path.join(TOKEN_SET_DIR, "tokenList.csv")
)


def calculated_properties():
    print("Writing the SMILES of trian set...")
    moses_dir = _moses_dir()
    with open(os.path.join(moses_dir, "train.csv"), encoding="utf-8") as file:
        with open(os.path.join(MOL_PROPERTIES_DIR, "LOGP", "train.csv"), "wb+") as logp_file:
            with open(os.path.join(MOL_PROPERTIES_DIR, "TPSA", "train.csv"), "wb+") as tpsa_file:
                for idx, line in enumerate(file):
                    if 1 <= idx:
                        try:
                            tokenizer.tokenize(line.strip())
                            mol = Chem.MolFromSmiles(line.strip())
                            calculate_list = ["MolLogP", "TPSA"]
                            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(calculate_list)
                            logp, tpsa = calculator.CalcDescriptors(mol)
                            print(f"\r\033[1;37;40m{line.strip()}\033[0m LOGP:%.4f, TPSA:%.4f"%(logp, tpsa), end="")
                            logp_file.write(f"{line.strip()},{round(logp, 4)}\n".encode("utf-8"))
                            tpsa_file.write(f"{line.strip()},{round(tpsa, 4)}\n".encode("utf-8"))
                        except:
                            pass

    print("Writing the SMILES of test set...")
    with open(os.path.join(moses_dir, "test.csv"), encoding="utf-8") as file:
        with open(os.path.join(MOL_PROPERTIES_DIR, "LOGP", "test.csv"), "wb+") as logp_file:
            with open(os.path.join(MOL_PROPERTIES_DIR, "TPSA", "test.csv"), "wb+") as tpsa_file:
                for idx, line in enumerate(file):
                    if 1 <= idx:
                        try:
                            mol = Chem.MolFromSmiles(line.strip())
                            calculate_list = ["MolLogP", "TPSA"]
                            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(calculate_list)
                            logp, tpsa = calculator.CalcDescriptors(mol)
                            print(f"\r\033[1;37;40m{line.strip()}\033[0m LOGP:{logp}, TPSA:{tpsa}", end="")
                            logp_file.write(f"{line.strip()},{round(logp, 4)}\n".encode("utf-8"))
                            tpsa_file.write(f"{line.strip()},{round(tpsa, 4)}\n".encode("utf-8"))
                        except:
                            pass


if not os.path.exists(os.path.join(MOL_PROPERTIES_DIR, "LOGP", "train.csv")):
    calculated_properties()


def calculated_synthesis_availability():
    SA = SaScore()
    print("Writing the SMILES of trian set...\n")
    moses_dir = _moses_dir()
    with open(os.path.join(moses_dir, "train.csv"), encoding="utf-8") as file:
        with open(os.path.join(MOL_PROPERTIES_DIR, "SA", "train.csv"), "wb+") as csv_file:
            for idx, line in enumerate(file):
                if 1 <= idx:
                    try:
                        tokenizer.tokenize(line.strip())
                        mol = Chem.MolFromSmiles(line.strip())
                        sa = SA(mol)
                        print(f"\r\033[1;37;40m{line.strip()}\033[0m Synthesis availability:%.4f"%(sa), end="")
                        csv_file.write(f"{line.strip()},{round(sa, 4)}\n".encode("utf-8"))
                    except:
                        pass

    print("Writing the SMILES of test set...\n")
    with open(os.path.join(moses_dir, "test.csv"), encoding="utf-8") as file:
        with open(os.path.join(MOL_PROPERTIES_DIR, "SA", "test.csv"), "wb+") as csv_file:
            for idx, line in enumerate(file):
                if 1 <= idx:
                    try:
                        tokenizer.tokenize(line.strip())
                        mol = Chem.MolFromSmiles(line.strip())
                        sa = SA(mol)
                        print(f"\r\033[1;37;40m{line.strip()}\033[0m Synthesis availability:%.4f"%(sa), end="")
                        csv_file.write(f"{line.strip()},{round(sa, 4)}\n".encode("utf-8"))
                    except:
                        pass


if not os.path.exists(os.path.join(MOL_PROPERTIES_DIR, "SA", "train.csv")):
    calculated_synthesis_availability()
