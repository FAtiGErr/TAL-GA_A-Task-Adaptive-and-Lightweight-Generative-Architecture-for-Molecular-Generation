class Tokenize:
    def __init__(self, dictPath = "tokenSet/Unigram.csv", tokenPath="tokenSet/tokenList.csv", forEmbedding=False):
        self.dictPath = dictPath

        self.vocabs_ = self.genDict()

        if tokenPath:
            vocabs_ = [i.strip().split(":") for i in open(tokenPath, encoding="utf-8").readlines()]
            vocabs_ = sorted(vocabs_, key=lambda x: eval(x[1]), reverse=True)
        else:
            vocabs_ = [(i.strip(), len(i.strip())) for i in open(dictPath, encoding="utf-8").readlines()]
            vocabs_ = sorted(vocabs_, key=lambda x: x[1], reverse=True)

        if forEmbedding:
            self.i2t = {i: j[0] for i, j in enumerate(vocabs_)}
            self.t2i = {j[0]: i for i, j in enumerate(vocabs_)}
            self.freq = [eval(i[1]) for i in vocabs_]
            self.t2f = {i[0]: eval(i[1]) for i in vocabs_}
        else:
            self.i2t = {i+1: j[0] for i, j in enumerate(vocabs_)}
            self.t2i = {j[0]: i+1 for i, j in enumerate(vocabs_)}

    def genDict(self):
        vocabs_ = [i.strip() for i in open(self.dictPath, encoding="utf-8").readlines()]
        self.max_len = max([len(i) for i in vocabs_])
        return vocabs_

    def tokenize(self, SMILES):
        return self.ReverseBMM(SMILES)

    def ReverseBMM(self, line):
        reverse = []
        while len(line) > 0:
            max_len = self.max_len
            if (len(line) < max_len):
                max_len = len(line)

            try_word = line[(len(line) - max_len):]
            while try_word not in self.vocabs_:
                if (len(try_word) == 1):
                    break
                try_word = try_word[1:]
            reverse.append(try_word)
            line = line[0:(len(line) - len(try_word))]
        reverse.reverse()
        return reverse

    def detokenize(self, arr):
        assert type(arr) is list, "Transform the arr into a list"
        try:
            return [self.i2t[i] for i in arr[:arr.index(0)]]
        except ValueError:
            return [self.i2t[i] for i in arr]
