import pandas as pd
import numpy as np


def embeddingMatrix(embeddingPath="weights/Word2vecEmbedding.csv"):
    df = pd.read_csv(embeddingPath, index_col="Token").values.tolist()
    embeddings = np.concatenate((np.zeros((1,100)), df), axis=0)
    return embeddings