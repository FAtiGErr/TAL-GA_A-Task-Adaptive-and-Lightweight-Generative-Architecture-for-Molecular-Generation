import os, gc
import random
import numpy as np
from tqdm import tqdm
from prepare import Tokenize
from embedding import embeddingMatrix
import tensorflow as tf
from tensorflow.keras import backend, Model, Input, metrics, losses, optimizers, layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        dll_path = os.path.join(conda_prefix, 'Library', 'bin')
        if os.path.exists(dll_path):
            os.add_dll_directory(dll_path)
            print(f"Added DLL directory: {dll_path}")
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], enable=True)
            print(f"Found GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU not found")
    return seed
seed = setup(233666)


class DatasetIterater(object):
    def __init__(self, dataname="mosesDs", label="train", max_len=60):
        Path = f"{dataname}/{label}.csv"
        self.maxlen = max_len
        dataset = []
        self.tokenizer = Tokenize()
        with open(Path, encoding="utf-8") as file:
            for idx, line in tqdm(enumerate(file)):
                if 1 <= idx:
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

    def iterator(self, batch_size):
        while True:
            for i in range(0, len(self.dataset), batch_size):
                x = self.dataset[i: i + batch_size]
                yield tf.constant(x)

    def __getitem__(self, idx):
        return self._to_tensor(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class Sampling(layers.Layer):
    """
    Uses (z_mean, z_log_var) to sample z.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = backend.random_normal(shape=(batch, dim), seed=seed)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class CNNEncoder(Model):
    def __init__(self, token_dim=100, latent_dim=100, max_len=60, token_num=1525, verbose=True):
        super(CNNEncoder, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.token_num = token_num
        embedding_matrix = embeddingMatrix("embedding/weights/Word2vecEmbedding.csv")
        self.embedding = layers.Embedding(input_dim=embedding_matrix.shape[0], trainable=True, weights=[embedding_matrix],
                                          output_dim=embedding_matrix.shape[1], name="VAEembedding", input_length=max_len)
        self.convA = layers.Conv1D(filters=self.latent_dim, kernel_size=3,
                                   strides=2, padding="same", activation="relu")
        self.convB = layers.Conv1D(filters=self.latent_dim, kernel_size=3,
                                   strides=2, padding="same", activation="relu")
        self.convC = layers.Conv1D(filters=self.latent_dim, kernel_size=3,
                                   strides=3, padding="same", activation="relu")
        self.MuStd_denseA = layers.Dense(units=self.latent_dim, activation="relu", use_bias=True, name="MuStd-DenseA")
        self.MuStd_denseB = layers.Dense(units=2*self.latent_dim, activation=None, use_bias=False, name="MuStd-DenseB")
        self._set_inputs(tf.TensorSpec([None, self.max_len], tf.int32, name="TokenInputs"))
        self.build(input_shape=(None, self.max_len))
        if verbose:
            self.model_description()

    def model_description(self):
        encoder_inputs = Input(shape=(self.max_len,), name="Token_Input")
        x = self.embedding(encoder_inputs)
        x = self.convA(x)
        x = self.convB(x)
        x = self.convC(x)
        x = layers.Flatten()(x)
        x = self.MuStd_denseA(x)
        x = self.MuStd_denseB(x)
        mu, std = tf.split(x, num_or_size_splits=2, axis=1)
        z = Sampling()(inputs=[mu, std])
        Model(encoder_inputs, z, name="CNNENCODER").summary()

    def call(self, x):
        x = self.embedding(x)
        x = self.convA(x)
        x = self.convB(x)
        x = self.convC(x)
        x = layers.Flatten()(x)
        x = self.MuStd_denseA(x)
        x = self.MuStd_denseB(x)
        mu, std = tf.split(x, num_or_size_splits=2, axis=1)
        z = Sampling()(inputs=[mu, std])
        return z, mu, std

    def save(self, filepath="model/CNNENCODER"):
        models.save_model(self, filepath)

    def load(self, filepath="model/CNNENCODER"):
        loaded = models.load_model(filepath)
        self.embedding = loaded.embedding
        self.convA = loaded.convA
        self.convB = loaded.convB
        self.convC = loaded.convC
        self.MuStd_denseA = loaded.MuStd_denseA
        self.MuStd_denseB = loaded.MuStd_denseB


class CNNDecoder(Model):
    def __init__(self, token_dim=100, latent_dim=200, max_len=60, token_num=1525, verbose=True):
        super(CNNDecoder, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.token_num = token_num

        self.latent_mapping = layers.Dense(latent_dim * self.max_len/2/2/3, name="LatentDense")
        self.latent_reshaping = layers.Reshape([int(self.max_len/2/2/3), self.latent_dim], name="LatentReshaping")
        self.deconvA = layers.Conv1DTranspose(filters=self.latent_dim, kernel_size=3, strides=3, padding="same",
                                              activation="relu", name="Conv1DTransposeA")
        self.deconvB = layers.Conv1DTranspose(filters=self.latent_dim, kernel_size=3, strides=2, padding="same",
                                              activation="relu", name="Conv1DTransposeB")
        self.deconvC = layers.Conv1DTranspose(filters=self.latent_dim, kernel_size=3, strides=2, padding="same",
                                              activation="relu", name="Conv1DTransposeC")
        self.filters_mapping = layers.Dense(units=self.latent_dim, name="ConvFiltersDense")
        self.out_prob = layers.TimeDistributed(layers.Dense(self.token_num, activation="softmax", name="TokenProbability"),
                                               name="TimeDistributedTokens")
        self._set_inputs(tf.TensorSpec([None, self.latent_dim], tf.float32, name="LatentInput"))
        self.build(input_shape=(None, self.latent_dim))
        if verbose:
            self.model_description()

    def model_description(self):
        state_input = Input(shape=(self.latent_dim,), name="Latent_Input")
        z_mapped = self.latent_mapping(state_input)
        z_reshaped = self.latent_reshaping(z_mapped)
        y = self.deconvA(z_reshaped)
        y = self.deconvB(y)
        y = self.deconvC(y)
        y = self.filters_mapping(y)
        y = self.out_prob(y)
        Model(state_input, y, name="CNNDecoder").summary()

    def call(self, x):
        z_mapped = self.latent_mapping(x)
        z_reshaped = self.latent_reshaping(z_mapped)
        y = self.deconvA(z_reshaped)
        y = self.deconvB(y)
        y = self.deconvC(y)
        y = self.filters_mapping(y)
        y = self.out_prob(y)
        return y

    def save(self, filepath="model/CNNDECODER"):
        models.save_model(self, filepath)

    def load(self, filepath="model/CNNDECODER"):
        loaded = models.load_model(filepath)
        self.latent_mapping = loaded.latent_mapping
        self.latent_reshaping = loaded.latent_reshaping
        self.deconvA = loaded.deconvA
        self.deconvB = loaded.deconvB
        self.deconvC = loaded.deconvC
        self.filters_mapping = loaded.filters_mapping
        self.out_prob = loaded.out_prob


class CNNVAE(Model):

    def __init__(self, token_dim=100, latent_dim=200, max_len=60, token_num=1525, verbose=True):
        super(CNNVAE, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.token_num = token_num

        self.encoder = CNNEncoder(self.token_dim, self.latent_dim, self.max_len, self.token_num, verbose)
        self.decoder = CNNDecoder(self.token_dim, self.latent_dim, self.max_len, self.token_num, verbose)
        print("Total parameters: %.4fM"%((self.encoder.count_params()+self.decoder.count_params())/1e6))

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.recon_loss_tracker = metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, x):
        with tf.GradientTape() as tape:
            z, z_mu, z_std = self.encoder(x)
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.reduce_sum(losses.sparse_categorical_crossentropy(x, recon)))
            kl_loss = - 0.5 * (1 + z_std - tf.square(z_mu) - tf.exp(z_std))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, x):
        z, z_mu, z_std = self.encoder(x)
        recon = self.decoder(z)
        recon_loss = tf.reduce_mean(tf.reduce_sum(losses.sparse_categorical_crossentropy(x, recon)))
        kl_loss = - 0.5 * (1 + z_std - tf.square(z_mu) - tf.exp(z_std))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return { "loss": self.total_loss_tracker.result(),
                 "recon_loss": self.recon_loss_tracker.result(),
                 "kl_loss": self.kl_loss_tracker.result()}

    def call(self, inputs):
        z, z_mean, z_log_var = self.encoder(inputs)
        recon = self.decoder(z)
        return recon

    def train(self, batch_size=256):
        print("...ABOUT TO TRAIN THE MODEL...")
        print("LOADING TRAINING DATA AND VALIDATION DATA...")
        self.trainDs = DatasetIterater(label="train")
        self.testDs = DatasetIterater(label="test")

        print("TRAINING STARTED.")
        lr_callback = ReduceLROnPlateau(monitor="val_recon_loss", factor=0.8, patience=3,
                                        verbose=1, min_lr=1e-6, mode="min", min_delta=1)
        stop_callback = EarlyStopping(monitor="val_recon_loss", patience=10,
                                      restore_best_weights=True, mode="min", min_delta=1)
        logs = f"model/LOGS/VAE_TB_logs/"
        tensor_board = TensorBoard(log_dir=logs,
                                   histogram_freq=200,
                                   embeddings_freq=200,
                                   update_freq="batch")
        self.compile(optimizer=optimizers.Adam(lr=1e-3))
        self.history = self.fit(x=self.trainDs.iterator(batch_size),
                                steps_per_epoch=int(len(self.trainDs) / batch_size), epochs=1000, verbose=1,
                                callbacks=[lr_callback, stop_callback, tensor_board],
                                use_multiprocessing=False,
                                validation_data=self.testDs.iterator(batch_size),
                                validation_steps=10,
                                workers=1)

    def save(self):
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.encoder.load()
        self.decoder.load()

    def save_embedding(self):
        weights = self.encoder.embedding.weights[0].numpy()
        import pandas as pd
        pd.DataFrame(weights, columns=list(range(weights.shape[1]))).to_csv(
            "embedding/weights/VAE_Refined_Embedding.csv", index=False)


if not os.path.exists("embedding/weights/VAE_Refined_Embedding.csv"):
    cnnvae = CNNVAE()
    cnnvae.train(500)
    cnnvae.save()
    cnnvae.save_embedding()