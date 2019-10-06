
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start of sentence
EOS_token = 2  # End of sentence

class Voc:
    """ This class is essentially a data structure that holds vocabulary
    information for the corpus plus some useful information like word counts,
    word indexes and a method for removing words that occur less than
    "min_count" times.
    """

    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        """ Adds all the words of a sentence in the Voc structure.
        :param sentence: string
        """

        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """ Add a word in the Voc structure
        :param word:
        """

        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """ Remove words below a certain count threshold
        :param min_count: int
        :return: Voc
        """

        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index),
            len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)
