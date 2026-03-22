import math
import os, gc
import random
import optuna
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from prepare import Tokenize
from embedding import embeddingMatrix
from optuna.samplers import RandomSampler
from optuna.pruners import HyperbandPruner
from vae import CNNVAE, CNNEncoder, CNNDecoder
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras import backend, Model, Input, metrics, losses, optimizers, layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import argparse
import csv
from config import (
    MOL_PROPERTIES_DIR,
    EMBEDDING_WEIGHTS_DIR,
    MODEL_DIR,
    MODEL_LOGS_DIR,
    set_working_directory,
)


set_working_directory()


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.set_soft_device_placement(True)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], enable=True)
        print(f"Found GPU: {gpus[0].name}")
    else:
        print("GPU not found, running on CPU.")
    return seed


seed = setup(233666)


class DatasetIterater(object):
    def __init__(self, dataname="LOGP", label="train", max_len=60, paramOpt=True):
        if paramOpt:
            data_path = os.path.join(MOL_PROPERTIES_DIR, dataname, f"ParamOpt-{label}.csv")
        else:
            data_path = os.path.join(MOL_PROPERTIES_DIR, dataname, f"{label}.csv")

        self.maxlen = max_len
        self.tokenizer = Tokenize()
        self.dataset = []

        with open(data_path, encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in tqdm(reader):
                if not row or len(row) < 2:
                    continue
                smi = row[0].strip()
                y_raw = row[-1].strip()
                try:
                    y = float(y_raw)
                except (TypeError, ValueError):
                    # Skip header or malformed rows.
                    continue

                tokens = self.tokenizer.tokenize(smi)
                if len(tokens) > self.maxlen:
                    continue

                try:
                    token_ids = [int(self.tokenizer.t2i[t]) for t in tokens]
                except KeyError:
                    continue

                token_ids += (self.maxlen - len(token_ids)) * [0]
                self.dataset.append([token_ids, y])

        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class DatasetSequence(Sequence):

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = zip(*batch)

        return np.array(x, dtype=np.int32), np.array(y, dtype=np.float32)


class DPCNNSignature(Model):
    def __init__(self,
                 token_dim=100,
                 latent_dim=100,
                 max_len=60,
                 verbose=True):
        super(DPCNNSignature, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        embedding_matrix = embeddingMatrix(os.path.join(EMBEDDING_WEIGHTS_DIR, "Word2vecEmbedding.csv"))
        self.embedding = layers.Embedding(input_dim=embedding_matrix.shape[0],output_dim=embedding_matrix.shape[1],
                                          weights=[embedding_matrix], trainable=True, name="DPCNNembedding",
                                          input_length=max_len, input_shape=(max_len,))
        self.regional_conv = layers.Conv2D(filters=self.latent_dim, kernel_size=(3, self.token_dim), strides=1, padding="valid",
                                           data_format="channels_last", name="RegionalConvolution", activation="relu")
        self.normal_conv = layers.Conv2D(filters=self.latent_dim, kernel_size=(3, 1), data_format="channels_last",
                                         strides=1, padding="same", name="NormalConvolution", activation="relu")
        self.max_pooling = layers.MaxPool2D(pool_size=(3, 1), strides=3, padding='same', name="MaxPooling")

        self.residual_add = layers.Add(name="ResidualAddOperation")

        self._set_inputs(tf.TensorSpec([None, self.max_len], tf.int32, name="TokenInputs"))
        self.build(input_shape=(None, self.max_len))
        if verbose:
            self.model_description()

    def residual(self, x):
        x_m = self.max_pooling(x)
        x_1 = self.normal_conv(x_m)
        x_2 = self.normal_conv(x_1)
        x = self.residual_add([x_m, x_2])
        return x

    def signature(self, x):
        x = self.embedding(x)
        x = tf.expand_dims(x, -1)
        x_re = self.regional_conv(x)
        x = self.normal_conv(x_re)
        x = self.normal_conv(x)
        x = layers.Add(name="RegionalAddOperation")([x_re, x])
        # while x.shape[1] > 1: x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        x = self.residual(x)
        return x

    def model_description(self):
        tokens_input = Input(shape=(self.max_len,), name="TokensInput")
        x = self.signature(tokens_input)
        x = layers.Flatten()(x)
        Model(tokens_input, x, name="DPCNN").summary()

    def call(self, x):
        x = self.signature(x)
        x = layers.Flatten()(x)
        return x

    def load(self, molProperty, qspr_model_type):
        filepath = os.path.join(MODEL_DIR, f"{qspr_model_type}-DPCNN({molProperty})")
        loaded = models.load_model(filepath)
        self.embedding = loaded.embedding
        self.regional_conv = loaded.regional_conv
        self.normal_conv = loaded.normal_conv


class SeqQSPR(Model):
    # noinspection PyUnusedLocal
    # TODO: TRY DEPRECATING THE VAE ENCODER
    def __init__(self,
                 token_dim=100,
                 latent_dim=200,
                 max_len=60,
                 token_num=1525,
                 molProperty=None,
                 qspr_kwargs=None,
                 verbose=True):
        super(SeqQSPR, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.token_num = token_num
        self.molProperty = molProperty
        if not qspr_kwargs:
            qspr_kwargs = {}

        self.dpcnn = DPCNNSignature(self.token_dim, self.latent_dim, self.max_len, verbose)
        print("Total parameters: %.4fM"%(self.dpcnn.count_params()/1e6))
        self.mlp = models.Sequential(name="SignatureFNN")
        self.mlp.add(Input(shape=(latent_dim), name="MergedSignatureInput"))
        acti = qspr_kwargs.get("activation", "relu")
        dropout = qspr_kwargs.get("dropout", 0.1)
        shrink = qspr_kwargs.get("shrink", 1)
        for i in range(qspr_kwargs.get("layers", 12)):
            units = int(2*self.latent_dim*shrink**(i-1))
            exec(f"self.mlp.add(layers.Dense(units, activation=acti, name='Dense{i+1}'))")
            exec(f"self.mlp.add(layers.Dropout(rate=dropout, name='Dropout{i+1}'))")
        self.mlp.add(layers.Dense(1, name="Output"))
        if verbose:
            self.mlp.summary()
        print("DPCNN and SeqQSPR-MLP initialized.")

        self._set_inputs(tf.TensorSpec([None, self.max_len], tf.int32, name="TokenInputs"))
        self.build(input_shape=(None, self.max_len))
        if verbose:
            self.model_description()

    def model_description(self):
        encoder_inputs = Input(shape=(self.max_len,), name="TokenInputs")
        z = self.dpcnn(encoder_inputs)
        x = self.mlp(z)
        Model(encoder_inputs, x, name="SeqQSPR").summary()

    def call(self, inputs):
        x = self.dpcnn(inputs)
        x = self.mlp(x)
        return x

    def saveModel(self):
        models.save_model(self.dpcnn, os.path.join(MODEL_DIR, f"SeqQSPR-DPCNN({self.molProperty})"))
        models.save_model(self.mlp, os.path.join(MODEL_DIR, f"SeqQSPR-MLP({self.molProperty})"))

    def load(self):
        self.dpcnn.load(self.molProperty, "SeqQSPR")
        self.mlp = models.load_model(os.path.join(MODEL_DIR, f"SeqQSPR-MLP({self.molProperty})"))
        print("DPCNN and SeqQSPR-MLP reloaded")

    def save_embedding(self):
        weights = self.dpcnn.embedding.weights[0].numpy()
        import pandas as pd
        pd.DataFrame(weights, columns=list(range(weights.shape[1]))).to_csv(
            os.path.join(EMBEDDING_WEIGHTS_DIR, f"SeqQSPR({self.molProperty})_Refined_Embedding.csv"), index=False)


def log_info(message):
    print(f"[DPCNN] {message}")


class ModelParamOptimize:
    def __init__(self, molProperty="LOGP", batch_size=200, max_len=60):
        self.molProperty = molProperty
        self.batch_size = batch_size
        self.max_len = max_len
        self.model = "SeqQSPR"
        self.trainDs = DatasetIterater(dataname=molProperty, label="train", max_len=60, paramOpt=True)
        self.testDs = DatasetIterater(dataname=molProperty, label="test", max_len=60, paramOpt=True)
        self.study = self.make_study()
        self.tolerance = 1e-2
        self.initial_lr = 1e-3

    def build_model(self, trial):
        params_dict = {"activation":trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"]),
                       "layers": trial.suggest_int("layers", low=2, high=8, step=1),
                       "dropout": trial.suggest_float("dropout", low=0, high=0.5, step=0.05),
                       "shrink": trial.suggest_float("shrink", low=0.5, high=1.0, step=0.05)}
        qspr = SeqQSPR(max_len=self.max_len, molProperty=self.molProperty, qspr_kwargs=params_dict, verbose=False)
        return qspr

    def param_opt(self, trial):
        tf.keras.backend.clear_session()
        model = self.build_model(trial)
        model.compile(optimizer=optimizers.Adam(self.initial_lr),
                      loss=losses.MSE,
                      metrics=[metrics.MeanAbsoluteError(name="MAE"),
                               metrics.RootMeanSquaredError(name="RMSE"),
                               metrics.MeanAbsolutePercentageError(name="MAPE")])

        pruning_callback = TFKerasPruningCallback(trial=trial, monitor="val_loss")

        stop_callback = EarlyStopping(monitor="val_loss",
                                      mode='min',
                                      min_delta=self.tolerance,
                                      restore_best_weights=True,
                                      patience=5)

        lr_reduce_callback = ReduceLROnPlateau(monitor="val_loss",
                                               mode='min',
                                               factor=0.1,
                                               min_delta=self.tolerance,
                                               patience=2)

        train_seq = DatasetSequence(self.trainDs.dataset, self.batch_size)
        test_seq = DatasetSequence(self.testDs.dataset, self.batch_size)

        history = model.fit(
            train_seq,
            epochs=100,
            validation_data=test_seq
        )


        evaluate = history.model.evaluate(test_seq)
        mae = evaluate[1]
        return mae

    def make_study(self):
        study_dir = os.path.join(MODEL_DIR, self.model, self.molProperty)
        os.makedirs(study_dir, exist_ok=True)
        self.SQL_path = f"sqlite:///{study_dir}/PARAMS.db"
        study = optuna.create_study(study_name=self.model,
                                    pruner=HyperbandPruner(),
                                    direction='minimize',
                                    storage=self.SQL_path,
                                    load_if_exists=True)
        return study

    def optimize(self, n_trials=3000):
        completed = len(self.study.trials)
        if completed >= n_trials:
            log_info(f"{self.molProperty}: optimization already complete ({completed}/{n_trials} trials).")
            return
        remaining = n_trials - completed
        log_info(f"{self.molProperty}: starting optimization ({completed}/{n_trials} done, {remaining} remaining).")
        self.study.optimize(self.param_opt, n_trials=remaining,
                            catch=(optuna.exceptions.StorageInternalError,
                                   optuna.exceptions.TrialPruned))
        log_info(f"{self.molProperty}: optimization finished ({len(self.study.trials)}/{n_trials} trials).")

    def save_best(self):
        log_info(f"{self.molProperty}: training final model with best hyperparameters.")
        out_dir = os.path.join(MODEL_DIR, self.model, self.molProperty)
        os.makedirs(out_dir, exist_ok=True)
        dataFramePath = f"{out_dir}/OptimizationProcess.csv"
        study_dataframe = self.study.trials_dataframe()
        study_dataframe.to_csv(dataFramePath)
        complete_trials = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not complete_trials:
            raise RuntimeError(
                f"{self.molProperty}: no completed Optuna trials with valid objective values."
            )
        best_trial = min(complete_trials, key=lambda t: t.value)
        param_best = best_trial.params

        self.trainDs = DatasetIterater(dataname=self.molProperty, label="train", max_len=60, paramOpt=False)
        self.testDs = DatasetIterater(dataname=self.molProperty, label="test", max_len=60, paramOpt=False)

        params_dict = {"activation":param_best["activation"], "layers": param_best["layers"],
                       "dropout": param_best["dropout"], "shrink": param_best["shrink"]}

        qspr = SeqQSPR(max_len=self.max_len, molProperty=self.molProperty, qspr_kwargs=params_dict, verbose=True)
        qspr.compile(optimizer=optimizers.Adam(self.initial_lr), loss=losses.MSE,
                     metrics=[metrics.MeanAbsoluteError(name="MAE"),
                              metrics.RootMeanSquaredError(name="RMSE"),
                              metrics.MeanAbsolutePercentageError(name="RMPE")])

        stop_callback = EarlyStopping(monitor="val_MAE",
                                      mode='min',
                                      min_delta=self.tolerance,
                                      restore_best_weights=True,
                                      patience=15)
        logs = os.path.join(MODEL_LOGS_DIR, f"{self.model}{self.molProperty}_TB_logs")
        tensor_board = TensorBoard(log_dir=logs,
                                   histogram_freq=200,
                                   embeddings_freq=200,
                                   update_freq="batch")
        lr_reduce_callback = ReduceLROnPlateau(monitor="val_loss", mode='min', factor=0.1, verbose=1,
                                               min_delta=self.tolerance, patience=3)
        train_seq = DatasetSequence(self.trainDs.dataset, self.batch_size)
        test_seq = DatasetSequence(self.testDs.dataset, self.batch_size)

        history = qspr.fit(
            train_seq,
            epochs=1000,
            verbose=1,
            callbacks=[stop_callback, lr_reduce_callback, tensor_board],
            validation_data=test_seq
        )

        qspr.saveModel()
        qspr.save_embedding()
        log_info(f"{self.molProperty}: model and embedding saved successfully.")
        return history


# Replace old auto-run block with explicit CLI entrypoint.
def parse_args():
    parser = argparse.ArgumentParser(description="Train SeqQSPR/DPCNN property predictors.")
    parser.add_argument("--properties", nargs="+", default=["TPSA", "LOGP", "SA"],
                        help="Property names to train, e.g. --properties CACO2")
    parser.add_argument("--trials", type=int, default=3000,
                        help="Target total Optuna trials for each property.")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if refined embedding exists.")
    return parser.parse_args()


def run_property(mp, target_trials=3000, force=False):
    embedding_path = os.path.join(EMBEDDING_WEIGHTS_DIR, f"SeqQSPR({mp})_Refined_Embedding.csv")
    if os.path.exists(embedding_path) and not force:
        log_info(f"{mp}: skip training because embedding already exists -> {embedding_path}")
        return

    log_info(f"{mp}: pipeline started.")
    opt = ModelParamOptimize(molProperty=mp)
    opt.optimize(n_trials=target_trials)
    opt.save_best()
    del opt
    gc.collect()
    log_info(f"{mp}: pipeline completed.")


def main():
    args = parse_args()
    props = [p.upper() for p in args.properties]
    log_info(f"Run started for properties: {', '.join(props)}")
    log_info(f"Target trials per property: {args.trials}")
    for mp in props:
        run_property(mp, target_trials=args.trials, force=args.force)
    log_info("All requested properties finished.")


if __name__ == "__main__":
    main()
