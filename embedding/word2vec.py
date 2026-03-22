import os
import time
import heapq
import itertools
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from prepare import Tokenize
from config import TOKEN_SET_DIR, MOSES_DS_DIR, WORD2VEC_EMBEDDING_FILE, set_working_directory


set_working_directory()

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(gpus[0], enable=True)
tokenizer = Tokenize(dictPath=os.path.join(TOKEN_SET_DIR, "Unigram.csv"),
                     tokenPath=os.path.join(TOKEN_SET_DIR, "tokenList.csv"),
                     forEmbedding=True)


# ======================================================================================================================
# ==================================================CBOW PARAMETERS=====================================================
# ======================================================================================================================
W2V_parameter = {
    'arch': 'skip_gram',  # Architecture: skip_gram or cbow
    'algm': 'negative_sampling',  # Training algorithm:  negative_sampling or hierarchical_softmax
    'window_size': 3,  # Num of words on the left or right side of target word within a window
    'batch_size': 256,
    'embedding_dim': 100,  # Length of word vector
    'neg_count': 10,  # Num of negative words to sample
    'power': 0.75,  # Distortion for negative sampling
    'epoch': 10,
    'alpha': 0.01,  # Initial learning rate
    'min_alpha': 0.0001,   # Final learning rate
    'add_bias': True,  # Whether to add bias term to dotproduct between syn0 and syn1 vectors.
    'log_per_steps': 500,  # Every `log_per_steps` steps to log the value of loss to be minimized.
}


# ======================================================================================================================
# =================================================BUILD HUFFMAN TREE===================================================
# ======================================================================================================================
class WordTokenizer():
    """
    Vanilla word tokenizer that spits out space-separated tokens from raw text string.
    Note for non-space separated languages, the corpus must be pre-tokenized such that tokens are space-delimited.
    """

    def __init__(self):
        global W2V_parameter
        self.token_path = os.path.join(TOKEN_SET_DIR, "tokenList.csv")
        train_file = os.path.join(MOSES_DS_DIR, "train.csv")
        test_file = os.path.join(MOSES_DS_DIR, "test.csv")

        with open(train_file) as corpus:
            self._corpus = []
            for idx, line in enumerate(tqdm(corpus, desc="Getting Corpus...")):
                if idx > 0:
                    try:
                        self._corpus.append([tokenizer.t2i[i] for i in tokenizer.tokenize(line.strip())])
                    except KeyError:
                        continue

        with open(test_file) as corpus:
            for idx, line in enumerate(tqdm(corpus, desc="Getting Corpus...")):
                if idx > 0:
                    try:
                        self._corpus.append([tokenizer.t2i[i] for i in tokenizer.tokenize(line.strip())])
                    except KeyError:
                        continue

        self._token2int = tokenizer.t2i
        self._int2token = tokenizer.i2t
        self._token2freq = tokenizer.t2f

        self._sample = 1e-3
        self._keep_probs = self.build_probs()
        W2V_parameter['token_count'] = len(self._token2int)

    def make_tokens_dict(self):
        global W2V_parameter
        tokens = pd.read_csv(self.token_path)['Token'].values.tolist()
        token2int = {token: ix for ix, token in enumerate(tokens)}
        int2token = {ix: token for ix, token in enumerate(tokens)}
        tokens_freq = pd.read_csv(self.token_path).values.tolist()
        tokens_freq = {i[1]: i[2] for i in tokens_freq}
        return token2int, int2token, tokens_freq

    def build_probs(self):
        """
        Has the side effect of setting the following attributes: for each token `token` we have
        keep_probs[index] = keep prob of `token` for subsampling
        Args:
          filenames: list of strings, holding names of text files.
        """
        raw_vocab = list(self.token_freq_dict.items())
        _all_tokens_count = sum(list(self.token_freq_dict.values()))
        keep_probs_ = {}
        for index, (token, count) in enumerate(raw_vocab):
            frac = count / float(_all_tokens_count)
            keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
            keep_prob = np.minimum(keep_prob, 1.0)
            keep_probs_[token] = keep_prob
        return keep_probs_

    @property
    def table_tokens(self):
        return list(self._int2token.values())

    @property
    def token_freq_dict(self):
        return self._token2freq

    @property
    def token_int_dict(self):
        return self._token2int

    @property
    def int_token_dict(self):
        return self._int2token

    @property
    def corpus(self):
        return self._corpus

    @property
    def keep_probs(self):
        return self._keep_probs


class Word2VecDatasetBuilder(object):

    """
    Builds a tf.data.Dataset instance that generates matrices holding word indices for training Word2Vec models.
    """

    def __init__(self,
                 arch=W2V_parameter['arch'],
                 algm=W2V_parameter['algm'],
                 epochs=W2V_parameter['epoch'],
                 batch_size=W2V_parameter['batch_size'],
                 window_size=W2V_parameter['window_size'],
                 tokenizer=None):
        """
        Args:
          batch_size: int scalar, the returned tensors in `get_tensor_dict` have shapes [batch_size, :].
          window_size: int scalar, num of words on the left or right side of target word within a window.
        """
        self._tokenizer = tokenizer
        self._arch = arch
        self._algm = algm
        self._epochs = epochs
        self._batch_size = batch_size
        self._window_size = window_size

        self._max_depth = None
        self._code_points = None

    def _build_binary_tree(self, token_counts):
        """
        Builds a Huffman tree for hierarchical softmax. Has the side effect of setting `max_depth`.
        Args:
            token_counts: list of int, holding word counts.
                          Index of each entry is the same as the word index into the vocabulary.
        Returns:
            codes_points: an int numpy array of shape [token_size, 2*max_depth+1] of each vocabulary word,
                          where each row holds the codes (0-1 binary values) padded to `max_depth`, and points
                          (non-leaf node indices) padded to `max_depth`.
                          The last entry is the true length of code and point (<= `max_depth`).
        """
        token_size = len(token_counts)
        heap = [[token_counts[i], i] for i in range(token_size)]
        # initialize the min-priority queue, which has length `token_size`
        heapq.heapify(heap)

        # insert `token_size` - 1 internal nodes, with vocab words as leaf nodes.
        for i in range(token_size - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, [min1[0] + min2[0], i + token_size, min1, min2])

        # At this point we have a len-1 heap, and `heap[0]` will be the root of the binary tree;
        # where internal nodes store: 1. key (frequency) 2. vocab index  3. left child 4. right child
        # and leaf nodes store 1. key (frequencey) 2. vocab index

        # Traverse the Huffman tree rooted at `heap[0]` in the order of Depth-First-Search. Each stack item stores the
        # 1. `node`  2. code of the `node` (list) 3. point of the `node` (list)
        # `point`: the list of vocab IDs of the internal nodes along the path from the root up to `node` (not included)
        # `code`: the list of labels (0 or 1) of the edges along the path from the root up to `node`
        # they are empty lists for the root node `heap[0]`
        node_list = []
        max_depth, stack = 0, [[heap[0], [], []]]  # stack: [root, code, point]
        while stack:
            node, code, point = stack.pop()
            if node[1] < token_size:
                # leaf node: len(node) == 2
                node.extend([code, point, len(point)])
                max_depth = np.maximum(len(code), max_depth)
                node_list.append(node)
            else:
                # internal node: len(node) == 4
                point = np.array(list(point) + [node[1] - token_size])
                stack.append([node[2], np.array(list(code) + [0]), point])
                stack.append([node[3], np.array(list(code) + [1]), point])

        # `len(node_list[i]) = 5`
        node_list = sorted(node_list, key=lambda items: items[1])
        # Stores the padded codes and points for each vocab word
        codes_points = np.zeros([token_size, max_depth * 2 + 1], dtype=np.int64)
        for i in range(len(node_list)):
            length = node_list[i][4]  # length of code or point
            codes_points[i, -1] = length
            codes_points[i, :length] = node_list[i][2]  # code
            codes_points[i, max_depth:max_depth + length] = node_list[i][3]  # point
        self._max_depth = max_depth
        return codes_points

    def build_dataset(self):
        """Generates tensor dict mapping from tensor names to tensors.

        Returns:
          dataset: a tf.data.Dataset instance, holding the a tuple of tensors
            (inputs, labels, progress)
            when arch=='skip_gram', algm=='negative_sampling'
              inputs: [N],                    labels: [N]
            when arch=='cbow', algm=='negative_sampling'
              inputs: [N, 2*window_size+1],   labels: [N]
            when arch=='skip_gram', algm=='hierarchical_softmax'
              inputs: [N],                    labels: [N, 2*max_depth+1]
            when arch=='cbow', algm=='hierarchical_softmax'
              inputs: [N, 2*window_size+1],   labels: [N, 2*max_depth+1]
            progress: [N], the percentage of sentences covered so far. Used to
              compute learning rate.
        """
        token_counts = list(self._tokenizer.token_freq_dict.values())
        keep_probs = list(self._tokenizer._keep_probs.values())

        if self._algm == 'hierarchical_softmax':
            codes_points = tf.constant(self._build_binary_tree(token_counts))
        elif self._algm == 'negative_sampling':
            codes_points = None
        else:
            raise ValueError('algm must be hierarchical_softmax or negative_sampling')

        keep_probs = tf.cast(tf.constant(keep_probs), 'float32')

        # corpus length times num of epochs
        num_smiles = len(self._tokenizer.corpus) * self._epochs

        def generator_fn():
            for _ in range(self._epochs):
                for smiles in self._tokenizer.corpus:
                    yield smiles

        # dataset: [([int], float)]
        dataset = tf.data.Dataset.zip((
            tf.data.Dataset.from_generator(generator_fn, tf.int64, [None]),  # The shape is none
            tf.data.Dataset.from_tensor_slices(tf.range(num_smiles) / num_smiles)))
        # dataset: [([int], float)]
        dataset = dataset.map(lambda indices, progress: (subsample(indices, keep_probs), progress))
        # dataset: [([int], float)]
        # Constrain the smiles to have at least 2 tokens
        dataset = dataset.filter(lambda indices, progress: tf.greater(tf.size(indices), 1))
        # dataset: [((None, None), float)]
        dataset = dataset.map(lambda indices, progress: (
                generate_instances(indices, self._arch, self._window_size, self._max_depth, codes_points), progress))
        # dataset: [((None, None)), (None,)]
        # replicate `progress` to size `tf.shape(instances)[:1]`
        dataset = dataset.map(lambda instances, progress: (instances, tf.fill(tf.shape(instances)[:1], progress)))

        dataset = dataset.flat_map(lambda instances, progress:
                                   # form a dataset by unstacking `instances` in the first dimension,
                                   tf.data.Dataset.from_tensor_slices((instances, progress)))

        # TODO ==========================================batch the dataset==============================================
        dataset = dataset.batch(self._batch_size, drop_remainder=True)

        def prepare_inputs_labels(tensor, progress):

            if self._arch == 'skip_gram':
                if self._algm == 'negative_sampling':
                    tensor.set_shape([self._batch_size, 2])
                else:
                    tensor.set_shape([self._batch_size, 2 * self._max_depth + 2])
                inputs = tensor[:, :1]
                labels = tensor[:, 1:]

            else:
                if self._algm == 'negative_sampling':
                    tensor.set_shape([self._batch_size, 2 * self._window_size + 2])
                else:
                    tensor.set_shape([self._batch_size, 2 * self._window_size + 2 * self._max_depth + 2])
                inputs = tensor[:, :2 * self._window_size + 1]
                labels = tensor[:, 2 * self._window_size + 1:]

            if self._arch == 'skip_gram':
                inputs = tf.squeeze(inputs, axis=1)
            if self._algm == 'negative_sampling':
                labels = tf.squeeze(labels, axis=1)
            progress = tf.cast(progress, 'float32')
            return inputs, labels, progress

        dataset = dataset.map(lambda tensor, progress: prepare_inputs_labels(tensor, progress))

        return dataset


def subsample(indices, keep_probs):
    """
    Applies subsampling on words in a smiles. Tokens with high frequencies have lower keep probs.
    Args:
      indices: rank-1 int tensor, the word indices within a smiles.
      keep_probs: rank-1 float tensor, the prob to drop the each vocabulary word.
    Returns:
      indices: rank-1 int tensor, the word indices within a sentence after subsampling.
    """
    keep_probs = tf.gather(keep_probs, indices)
    randvars = tf.random.uniform(tf.shape(keep_probs), 0, 1)
    indices = tf.boolean_mask(indices, tf.less(randvars, keep_probs))
    return indices


def generate_instances(indices,
                       arch,
                       window_size,
                       max_depth=None,
                       codes_points=None):
    """
    Generates matrices holding word indices to be passed to Word2Vec models for each sentence.
    The shape and contents of output matrices depends on the architecture ('skip_gram', 'cbow') and training algorithm
    ('negative_sampling', 'hierarchical_softmax').
    It takes a list of word indices in a subsampled-sentence as input, where ** each word is a target word **, and
    their context words are those within the window centered at a target word. For SKIPGRAM architecture,
    `num_context_words` instances are generated for a target word, and for CBOW architecture, a single instance is
    generated for a target word.

    If `codes_points` is not None (for 'hierarchical softmax'), the word to be predicted (context word for
    'skip_gram', and target word for 'cbow') are represented by their 'codes' and 'points' in the Huffman tree
    (See `_build_binary_tree`).

    Args:
        indices: rank-1 int tensor, the word indices within a sentence after subsampling.
        arch: scalar string, architecture ('skip_gram' or 'cbow').
        window_size: int scalar, num of words on the left or right side of target word within a window.
        max_depth: (Optional) int scalar, the max depth of the Huffman tree.
        codes_points: (Optional) an int tensor of shape [vocab_size, 2 * max_depth + 1] of each token, where each row
                      holds the codes (0-1 binary values) padded to `max_depth`, and points (non-leaf node indices)
                      padded toc `max_depth`, The last entry is the true length of code and point (<= `max_depth`).

    Returns:
      instances: an int tensor holding word indices, with shape being
        when arch=='skip_gram', algm=='negative_sampling'
          shape: [N, 2]
        when arch=='cbow', algm=='negative_sampling'
          shape: [N, 2 * window_size + 2]
        when arch=='skip_gram', algm=='hierarchical_softmax'
          shape: [N, 2 * max_depth + 2]
        when arch=='cbow', algm='hierarchical_softmax'
          shape: [N, 2 * window_size + 2 * max_depth + 2]
    """

    def per_target_fn(index,
                      init_array):
        """
        Generate inputs and labels for each target word. 'index` is the index of the target word in `indices`.
        """
        reduced_size = tf.random.uniform([], maxval=window_size, dtype='int32')
        left = tf.range(tf.maximum(index - window_size + reduced_size, 0), index)
        right = tf.range(index + 1, tf.minimum(index + 1 + window_size - reduced_size, tf.size(indices)))
        context = tf.concat([left, right], axis=0)
        context = tf.gather(indices, context)

        if arch == 'skip_gram':
            # replicate `indices[index]` to match the size of `context` [N, 2]
            window = tf.stack([tf.fill(tf.shape(context), indices[index]), context], axis=1)
        elif arch == 'cbow':
            true_size = tf.size(context)
            # pad `context` to length `2 * window_size` with int 0
            window = tf.concat([tf.pad(context, [[0, 2 * window_size - true_size]]),
                                [true_size, indices[index]]], axis=0)
            # [1, 2 * window_size + 2]
            window = tf.expand_dims(window, axis=0)
        else:
            raise ValueError('architecture must be skip_gram or cbow.')

        if codes_points is not None:
            # [N, 2 * max_depth + 2] or [1, 2 * window_size + 2 * max_depth + 2]
            window = tf.concat([window[:, :-1],
                                tf.gather(codes_points, window[:, -1])], axis=1)
        return index + 1, init_array.write(index, window)

    size = tf.size(indices)
    # initialize a tensor array of length `tf.size(indices)`
    init_array = tf.TensorArray('int64', size=size, infer_shape=False)
    _, result_array = tf.while_loop(lambda i, ta: i < size, per_target_fn, [0, init_array], back_prop=False)
    instances = tf.cast(result_array.concat(), 'int64')
    if arch == 'skip_gram':
        if max_depth is None:
            instances.set_shape([None, 2])
        else:
            instances.set_shape([None, 2 * max_depth + 2])
    else:
        if max_depth is None:
            instances.set_shape([None, 2 * window_size + 2])
        else:
            instances.set_shape([None, 2 * window_size + 2 * max_depth + 2])

    return instances


# ======================================================================================================================
# =================================================UTILITY   FUNCTION===================================================
# ======================================================================================================================


def get_train_step_signature(arch,
                             algm,
                             batch_size,
                             window_size=None,
                             max_depth=None):
    """
    Get the training step signatures for `inputs`, `labels` and `progress` tensor.
    Args:
        arch: string scalar, architecture ('skip_gram' or 'cbow').
        algm: string scalar, training algorithm ('negative_sampling' or 'hierarchical_softmax').
    Returns:
        train_step_signature: a list of three tf.TensorSpec instances, specifying the tensor spec (shape and dtype) for
        `inputs`, `labels` and `progress`.
    """

    if arch == 'skip_gram':
        inputs_spec = tf.TensorSpec(shape=(batch_size,), dtype='int64')
    elif arch == 'cbow':
        inputs_spec = tf.TensorSpec(shape=(batch_size, 2 * window_size + 1), dtype='int64')
    else:
        raise ValueError('`arch` must be either "skip_gram" or "cbow".')

    if algm == 'negative_sampling':
        labels_spec = tf.TensorSpec(shape=(batch_size,), dtype='int64')
    elif algm == 'hierarchical_softmax':
        labels_spec = tf.TensorSpec(shape=(batch_size, 2 * max_depth + 1), dtype='int64')
    else:
        raise ValueError('`algm` must be either "negative_sampling" or "hierarchical_softmax".')

    progress_spec = tf.TensorSpec(shape=(batch_size,), dtype='float32')

    train_step_signature = [inputs_spec, labels_spec, progress_spec]

    return train_step_signature


# ======================================================================================================================
# =================================================MODEL CONSTRUCTUION==================================================
# ======================================================================================================================
class Word2VecModel(tf.keras.Model):
    def __init__(self,
                 arch=W2V_parameter['arch'],
                 algm=W2V_parameter['algm'],
                 hidden_size=W2V_parameter['embedding_dim'],
                 batch_size=W2V_parameter['batch_size'],
                 negatives=W2V_parameter['neg_count'],
                 power=W2V_parameter['power'],
                 alpha=W2V_parameter['alpha'],
                 min_alpha=W2V_parameter['min_alpha'],
                 add_bias=W2V_parameter['add_bias'],
                 random_seed=10,
                 tokenizer=None):
        """
        Args:
          token_counts: a list of ints, the counts of word tokens in the corpus.
          arch: string scalar, architecture ('skip_gram' or 'cbow').
          algm: string scalar, training algorithm ('negative_sampling' or 'hierarchical_softmax').
          hidden_size: int scalar, length of word vector.
          batch_size: int scalar, batch size.
          negatives: int scalar, num of negative words to sample.
          power: float scalar, distortion for negative sampling.
          alpha: float scalar, initial learning rate.
          min_alpha: float scalar, final learning rate.
          add_bias: bool scalar, whether to add bias term to dotproduct between syn0 and syn1 vectors.
          random_seed: int scalar, random_seed.
        """
        super(Word2VecModel, self).__init__()
        self._tokenizer = tokenizer
        self._token_counts = list(self._tokenizer.token_freq_dict.values())
        self._arch = arch
        self._algm = algm
        self._hidden_size = hidden_size
        self._token_size = len(self._token_counts)
        self._batch_size = batch_size
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._add_bias = add_bias
        self._random_seed = random_seed

        self._input_size = (self._token_size if self._algm == 'negative_sampling' else self._token_size - 1)

        self.add_weight('syn0', shape=[self._token_size, self._hidden_size],
                        initializer=tf.keras.initializers.RandomUniform(minval=-0.5 / self._hidden_size,
                                                                        maxval=0.5 / self._hidden_size))

        self.add_weight('syn1', shape=[self._input_size, self._hidden_size],
                        initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1))

        self.add_weight('biases', shape=[self._input_size], initializer=tf.keras.initializers.Zeros())

    def call(self, inputs, labels):
        """
        Runs the forward pass to compute loss.
        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or [batch_size, 2 * window_size + 1] (cbow)
            labels: int tensor of shape [batch_size] (negative_sampling) or [batch_size, 2 * max_depth + 1]
                    (hierarchical_softmax)
        Returns:
          loss: float tensor, cross entropy loss.
        """
        if self._algm == 'negative_sampling':
            loss = self._negative_sampling_loss(inputs, labels)
        elif self._algm == 'hierarchical_softmax':
            loss = self._hierarchical_softmax_loss(inputs, labels)
        return loss

    def _negative_sampling_loss(self, inputs, labels):
        """
        Builds the loss for negative sampling.
        Args:
            inputs: int tensor of shape [batch_size] (skip_gram) or [batch_size, 2 * window_size + 1] (cbow)
            labels: int tensor of shape [batch_size]
        Returns:
            loss: float tensor of shape [batch_size, negatives + 1].
        """
        _, syn1, biases = self.weights

        sampled_values = tf.nn.fixed_unigram_candidate_sampler(true_classes=tf.expand_dims(labels, axis=1),
                                                               num_true=1,
                                                               num_sampled=self._batch_size * self._negatives,
                                                               unique=False,  # Sampling without replacement if True
                                                               range_max=len(self._token_counts),
                                                               distortion=self._power,
                                                               unigrams=self._token_counts,
                                                               seed=self._random_seed)

        sampled = sampled_values.sampled_candidates
        sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
        inputs_syn0 = self._get_inputs_syn0(inputs)  # [batch_size, hidden_size]
        true_syn1 = tf.gather(syn1, labels)  # [batch_size, hidden_size]
        sampled_syn1 = tf.gather(syn1, sampled_mat)  # [batch_size, negatives, hidden_size]
        true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), axis=1)  # [batch_size]
        # [batch_size, negatives]
        sampled_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1), tf.transpose(sampled_syn1, (0, 2, 1)))

        if self._add_bias:
            true_logits += tf.gather(biases, labels)  # [batch_size]
            sampled_logits += tf.gather(biases, sampled_mat)  # [batch_size, negatives]

        # [batch_size]
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits),
                                                                     logits=true_logits)
        # [batch_size, negatives]
        sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits),
                                                                        logits=sampled_logits)

        loss = tf.concat(
            [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], axis=1)
        return loss

    def _hierarchical_softmax_loss(self, inputs, labels):
        """
        Builds the loss for hierarchical softmax.
        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or [batch_size, 2*window_size+1] (cbow)
          labels: int tensor of shape [batch_size, 2 * max_depth + 1]
        Returns:
          loss: float tensor of shape [sum_of_code_len]
        """
        _, syn1, biases = self.weights

        inputs_syn0_list = tf.unstack(self._get_inputs_syn0(inputs))
        codes_points_list = tf.unstack(labels)
        max_depth = (labels.shape.as_list()[1] - 1) // 2
        loss = []
        for i in range(self._batch_size):
            inputs_syn0 = inputs_syn0_list[i]  # [hidden_size]
            codes_points = codes_points_list[i]  # [2 * max_depth + 1]
            true_size = codes_points[-1]

            codes = codes_points[:true_size]
            points = codes_points[max_depth:max_depth + true_size]
            logits = tf.reduce_sum(
                tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
            if self._add_bias:
                logits += tf.gather(biases, points)

            # [true_size]
            loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.cast(codes, 'float32'), logits=logits))
        loss = tf.concat(loss, axis=0)
        return loss

    def _get_inputs_syn0(self, inputs):
        """
        Builds the activations of hidden layer given input words embeddings `syn0` and input word indices.
        Args:
          inputs: int tensor of shape [batch_size] (skip_gram) or [batch_size, 2*window_size+1] (cbow)
        Returns:
          inputs_syn0: [batch_size, hidden_size]
        """

        # syn0: [vocab_size, hidden_size]
        syn0, _, _ = self.weights

        if self._arch == 'skip_gram':
            inputs_syn0 = tf.gather(syn0, inputs)  # [batch_size, hidden_size]
        else:
            inputs_syn0 = []
            contexts_list = tf.unstack(inputs)
            for i in range(self._batch_size):
                contexts = contexts_list[i]
                context_words = contexts[:-1]
                true_size = contexts[-1]
                inputs_syn0.append(tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
            inputs_syn0 = tf.stack(inputs_syn0)

        return inputs_syn0


# ======================================================================================================================
# ==================================================RUN TRAINING MODEL==================================================
# ======================================================================================================================
def Train():
    """
    Train a word2vec model to obtain word embedding vectors.
    There are a total of four combination of architectures and training algorithms
    that the model can be trained with:
    architecture:
      - skip_gram
      - cbow (continuous bag-of-words)
    training algorithm
      - negative_sampling
      - hierarchical_softmax
    """
    tokenizer = WordTokenizer()
    word2vec = Word2VecModel(tokenizer=tokenizer)
    builder = Word2VecDatasetBuilder(tokenizer=tokenizer)
    dataset = builder.build_dataset()

    train_step_signature = get_train_step_signature(arch=W2V_parameter['arch'],
                                                    algm=W2V_parameter['algm'],
                                                    batch_size=W2V_parameter['batch_size'],
                                                    window_size=W2V_parameter['window_size'],
                                                    max_depth=builder._max_depth)

    optimizer = tf.keras.optimizers.SGD(1.0)
    print("Training started.")
    print("W2V_parameters updated.")

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, labels, progress):
        loss = word2vec(inputs, labels)
        gradients = tf.gradients(loss, word2vec.trainable_variables)
        learning_rate = tf.maximum(W2V_parameter['alpha'] * (1 - progress[0]) +
                                   W2V_parameter['min_alpha'] * progress[0], W2V_parameter['min_alpha'])

        if hasattr(gradients[0], '_values'):
            gradients[0]._values *= learning_rate
        else:
            gradients[0] *= learning_rate

        if hasattr(gradients[1], '_values'):
            gradients[1]._values *= learning_rate
        else:
            gradients[1] *= learning_rate

        if hasattr(gradients[2], '_values'):
            gradients[2]._values *= learning_rate
        else:
            gradients[2] *= learning_rate

        optimizer.apply_gradients(zip(gradients, word2vec.trainable_variables))

        return loss, learning_rate

    average_loss = 0.

    for step, (inputs, labels, progress) in enumerate(tqdm(dataset)):
        loss, learning_rate = train_step(inputs, labels, progress)
        average_loss += loss.numpy().mean()
        if step % W2V_parameter['log_per_steps'] == 0:
            if step > 0:
                average_loss /= W2V_parameter['log_per_steps']
            print('step:', step, 'average_loss: %f' % average_loss, 'learning_rate:', learning_rate.numpy())
            average_loss = 0.

    syn0_final = word2vec.weights[0].numpy()
    df = pd.DataFrame(syn0_final, index=list(word2vec._tokenizer.int_token_dict.values()))
    os.makedirs(os.path.dirname(WORD2VEC_EMBEDDING_FILE), exist_ok=True)
    df.to_csv(WORD2VEC_EMBEDDING_FILE)


if not os.path.exists(WORD2VEC_EMBEDDING_FILE):
    Train()