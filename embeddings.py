import numpy as np
import logging

from gensim.models.keyedvectors import KeyedVectors

log = logging.getLogger(__name__)


class EmbeddingsLoader:
    """ Local Embeddings Loader """

    def __init__(self):
        """ Initialization """

        self.dim = None
        self.vocab_size = None
        self.vocab = None
        self.mode = None
        self.vectors = None

    def load(self, embeddings_file, mode='word2vec'):
        """ Load an embeddings file """
        print(f'Loading embeddings from {embeddings_file} ...')
        print('Embeddings file: ', embeddings_file)
        extension = embeddings_file.split('.')[-1]
        print('Split embeddings file...')

        if extension == 'vec':
            print('This is a vectorized embeddings file...')

            vectors = KeyedVectors.load_word2vec_format(embeddings_file,
                                                        binary=False)
        elif extension == 'bin':
            print('This is a binary embeddings file...')
            if mode == 'word2vec':
                print('WORD2VEC')
                vectors = KeyedVectors.load_word2vec_format(embeddings_file,
                                                            binary=True)
                print('Vectors: ', vectors)
            elif mode == 'fasttext':
                # vectors = FastText.load_fasttext_format(embeddings_file)
                vectors = KeyedVectors.load_word2vec_format(embeddings_file,
                                                            binary=True,
                                                            encoding='utf-8',
                                                            unicode_errors='ignore')
            else:
                log.info('Provide an acceptable mode, word2vec or fasttext')
                return
        else:
            log.info('Provide a .vec or a .bin file')
            print('Provide a .vec or a .bin file')
            return

        self.vectors = vectors
        self.vocab = list(vectors.vocab.keys())
        self.dim = vectors.vector_size
        self.vocab_size = len(self.vocab)

    def get_word_vectors(self, utterance, separator=' '):
        """
        Fetch embeddings for all the tokens in the utterance
        Return a dictionary mapping the tokens to their vector representations
        """
        word_vectors = {}
        for token in set(utterance.split(separator)):
            if token in self.vectors:
                word_vectors[token] = self.vectors[token]

        return word_vectors

    def get_centroid_vector(self, utterance):
        """
        Fetch embeddings for all the tokens in the utterance
        Return the centroid vector as an utterance representation
        """
        word_vectors = self.get_word_vectors(utterance)
        features = np.zeros(self.dim)
        tokens = utterance.split()

        for token in tokens:
            if token in word_vectors:
                features = np.add(features, word_vectors[token])

        return features / len(tokens)

    def get_embedding_matrix(self, utterance, zero_padding=False,
                             max_length=10):
        """
        Fetch embeddings for all the tokens in the utterance
        Create a 2d matrix where the row at index i contains the embedding of
        the ith token
        Matrix shape : NxD where N the number of tokens in the utterance
        and D the dimensionality of the embeddings
        """
        word_vectors = self.get_word_vectors(utterance)
        tokens = utterance.split()
        features = np.zeros(shape=(len(tokens), self.dim))

        for i in range(len(tokens)):
            if tokens[i] in word_vectors:
                features[i, :] = word_vectors[tokens[i]]
            else:
                log.warning(
                    f'Could not find embedding vector '
                    f'for token: [{tokens[i]}]'
                )

        # Perform zero padding
        if zero_padding:
            if max_length and max_length > len(tokens):
                return np.pad(features, ((0, max_length - len(tokens)), (0, 0)),
                              'constant')

        return features

    def get_keras_initializer(self, model_token_vocab):
        """
        Get a numpy array as an initializer for a Keras Embedding layer
        model_token_vocab is a dictionary with tokens as keys and indexes as
        values
        """
        initializer = np.zeros(shape=(len(model_token_vocab), self.dim))
        for token, index in model_token_vocab.items():
            try:
                initializer[index, :] = self.vectors[token]
            except KeyError:
                initializer[index, :] = np.random.random(self.dim)

        return initializer
