import os

# Project root (directory where this config file is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Common project directories
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_LOGS_DIR = os.path.join(MODEL_DIR, "LOGS")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PSO_RESULTS_DIR = os.path.join(RESULTS_DIR, "pso")
RESULTS_MODELS_DIR = os.path.join(RESULTS_DIR, "models")
RESULTS_MODELS_QSPR_DIR = os.path.join(RESULTS_MODELS_DIR, "QSPR")
RESULTS_MODELS_VAE_DIR = os.path.join(RESULTS_MODELS_DIR, "VAE")
RESULTS_MODELS_VAE_WRONG_DIR = os.path.join(RESULTS_MODELS_VAE_DIR, "Wrongly Decoded")
RESULTS_FEATURE_DIR = os.path.join(RESULTS_DIR, "feature")
RESULTS_EXAMPLE_MOLS_DIR = os.path.join(RESULTS_DIR, "example-Mols")
RESULTS_LATENTS_DIR = os.path.join(RESULTS_DIR, "latents")
RESULTS_TOKENS_DIR = os.path.join(RESULTS_DIR, "tokens")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embedding")
EMBEDDING_WEIGHTS_DIR = os.path.join(EMBEDDING_DIR, "weights")
WORD2VEC_EMBEDDING_FILE = os.path.join(EMBEDDING_WEIGHTS_DIR, "Word2vecEmbedding.csv")
MOL_PROPERTIES_DIR = os.path.join(PROJECT_ROOT, "molProperties")
TOKEN_SET_DIR = os.path.join(PROJECT_ROOT, "tokenSet")
JOINT_CORPUS_DIR = os.path.join(PROJECT_ROOT, "jointCorpus")
MOSES_DS_DIR = os.path.join(PROJECT_ROOT, "mosesDs")
MOSES_DIR = os.path.join(PROJECT_ROOT, "moses")
PYTDC_DIR = os.path.join(PROJECT_ROOT, "pyTDC")
PREPARE_DIR = os.path.join(PROJECT_ROOT, "prepare")
FRAGSCORE_FILE = os.path.join(PREPARE_DIR, "fragscore.pkl.gz")


def project_path(*parts):
    """Build an absolute path under PROJECT_ROOT."""
    return os.path.join(PROJECT_ROOT, *parts)


def ensure_dir(path):
    """Create a directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def set_working_directory():
    """Set current working directory to PROJECT_ROOT."""
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT

