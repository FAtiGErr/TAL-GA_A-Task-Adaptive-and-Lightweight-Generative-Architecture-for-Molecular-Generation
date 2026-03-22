import os
import sentencepiece as sp
from config import JOINT_CORPUS_DIR, TOKEN_SET_DIR, set_working_directory


set_working_directory()


def raw_tokenizer():
    corpus = os.path.join(JOINT_CORPUS_DIR, "SMILESCorpus.csv")

    sp.SentencePieceTrainer.train_data(
        input=corpus,
        input_format="text",
        model_prefix=os.path.join(TOKEN_SET_DIR, "rawUnigram"),
        character_coverage=1.0,
        vocab_size=30000,
        num_sub_iterations=10,
        model_type="unigram",
        split_by_number=False,
        split_by_unicode_script=False,
        split_digits=False,
        add_dummy_prefix=False,
        hard_vocab_limit=False,
        train_extremely_large_corpus=True)


if not os.path.exists(os.path.join(TOKEN_SET_DIR, "rawUnigram.model")):
    raw_tokenizer()
