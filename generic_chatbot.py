"""A simple generic chatbot implementation
Code adapted from various sources by Dimitris Mavroeidis.
This script trains a Neural Network with sample bot discussions. It then runs
an evaluation phase where the user asks a question and the chatbot answers it.
It is really a toy project that is meant to check the understanding of how
NNs work and the quality of the code.
This work has been adapted from the pytorch chatbot tutorial
(https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).
It also used the method for creating and loading pre-trained embeddings from
the following tutorial:
https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
This project has been uploaded to github as a private repo for convenience.
https://github.com/dmavroeidis/generic_chatbot

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import csv
import datetime
import itertools
import os
import pickle
import random
import re
import unicodedata
from io import open

import bcolz
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils import checkpoint

from data import Voc, PAD_token, EOS_token, SOS_token
from model import EncoderRNN, LuongAttentionDecoderRNN
from decoder import GreedySearchDecoder

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("Processing device: ", device)

corpus_name = "metalwoz-v1"
corpus = os.path.join("data", corpus_name, "dialogues")
print("\ncorpus: \n", corpus)

# Load embeddings
glove_path = 'data/glove.6B'
EMBEDDING_DIMENSION = 300  # Can be 300, 200, 100, 50
vectors = bcolz.open(f'{glove_path}/6B.{str(EMBEDDING_DIMENSION)}.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.'
                         f'{str(EMBEDDING_DIMENSION)}_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.'
                            f'{str(EMBEDDING_DIMENSION)}_idx.pkl', 'rb'))

MAX_LENGTH = 10  # Maximum sentence length
MIN_COUNT = 1  # Minimum count to expel a word from the vocabulary. default=3


# Define the spell-checking object
# spell = SpellChecker(distance=2)
# spell.word_frequency.load_text_file('data/extra_vocabulary.txt', 'UTF-8')


def printLines(file, n=10):
    """ Print n lines of a file

    :param file:
    :param n:
    """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def loadLines(file_name):
    """  Reads an input file and returns a dictionary with:
    Keys: 'dialogId-[x]' where x is the number of each pair.
    Values: a list containing 2 utterances: The first one is the user's
    utterance and the second one is the system's response.

    :param file_name: str
    :return: list of lists
    """
    conversations_list = []
    with open(file_name, 'r', encoding='iso-8859-1') as f:
        csv_reader = csv.reader(f, delimiter=':')
        row_counter = 0
        print('File: ', file_name)
        for row in csv_reader:
            utterances = '\t'.join(row[-1].split('", "'))[3:-3]
            values = []
            utterance_counter = 0
            for utterance in utterances.split('\t'):
                if "Hello how may I help you?" not in utterance:
                    values.append(utterance)
                    if not utterance_counter % 2:
                        conversations_list.append(values)
                        values = []
                utterance_counter += 1
            row_counter += 1
        # print('\n\nconversations list: ' + str(conversations_list))
    return conversations_list


# Checks and corrects spelling errors in the list of conversations
# TODO: Change something the format of the output to work.
# def spell_checker(conversations):
#     new_conversations = []
#     for pair in conversations:
#         # print('PAIR: ', pair)
#         new_pair = []
#         for utterance in pair:
#             # print('UTTERANCE: ', utterance)
#             new_utterance = []
#             for word in utterance.split(' '):
#                 # print('WORD: ', word)
#                 try:
#                     if not len(spell.unknown([word])) == 0:
#                         print('Found misspelled word: ', word)
#                         word = spell.correction(word)
#                         print('Corrected to: ', word)
#                 except Exception as e:
#                     print("Some kind of error:")
#                     print(str(e))
#
#                 new_utterance.append(word)
#             new_pair.append(new_utterance)
#         new_conversations.append(new_pair)
#
#     return new_conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation[
                               "lines"]) - 1):  # We ignore the last line (no
            # answer for it)
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i + 1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


# def load_embeddings(embeddings_file, embeddings_mode='word2vec'):
#     """ Load embeddings """
#     print('Start loading embeddings...')
#     embeddings = EmbeddingsLoader()
#     embeddings.load(embeddings_file, mode=embeddings_mode)
#
#     # self.logsystem.log_info(f'Loading embeddings.. from {embeddings_file}')
#
#     loginfo = PrettyTable(['Vocabulary size', 'Dimensionality'])
#     loginfo.add_row([embeddings.vocab_size, embeddings.dim])
#     # self.logsystem.log_info('\n' + loginfo.get_string(title='Loaded
#     Embeddings info'))
#
#     del loginfo, embeddings_mode, embeddings_file
#     return embeddings


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    """

    :param s:
    :return:
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


# Read query/response pairs and return a voc object
def readVocs(data_file, corpus_name):
    """

    :param data_file:
    :param corpus_name:
    :return:
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(data_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    new_lines = []

    # Remove training lines that do not contain the TABs.
    tab_counter = 0
    for line in lines:
        if '\t' in line:
            new_lines.append(line)
        else:
            tab_counter += 1
    print("Removed ", tab_counter, " lines of total ", len(lines),
          "that didn't contain a TAB.")
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in new_lines]
    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(p):
    """ Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH
    threshold
    :param p: list
    :return: boolean
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(
        p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    """ Filter pairs using filterPair condition

    :param pairs: list of lists
    :return: list of lists
    """
    return [pair for pair in pairs if filter_pair(pair)]


# Using the functions defined above, return a populated voc object and pairs
# list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    """

    :param corpus:
    :param corpus_name:
    :param datafile:
    :param save_dir:
    :return:
    """
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs


def trimRareWords(voc, pairs, minimum_count):
    """ Words that appear less frequently than minimum_count are removed
    from the vocabulary.

    :param voc: Voc
    :param pairs: list of lists
    :param minimum_count: int
    :return: list of lists
    """

    voc.trim(minimum_count)
    # Remove pairs that contain trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(
        len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


# Returns the indexes of words for an input sentence
def indexesFromSentence(voc, sentence):
    """

    :param voc:
    :param sentence:
    :return:
    """
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


# If the sentence has length less than MAX_LENGHT, pad it with zeroes
def zeroPadding(l, fillvalue=PAD_token):
    """

    :param l:
    :param fillvalue:
    :return:
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# Create binary matrix. This is the same as the input tensor, but has '1'
# instead of the word index in the place of the word and 0 elsewhere.
def binaryMatrix(l, value=PAD_token):
    """

    :param l:
    :param value:
    :return:
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    """

    :param l:
    :param voc:
    :return:
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    """

    :param l:
    :param voc:
    :return:
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zeroPadding(indexes_batch)
    mask = binaryMatrix(pad_list)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(pad_list)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    """

    :param voc:
    :param pair_batch:
    :return:
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_length = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_length


def maskNLLLoss(inp, target, mask):
    """

    :param inp:
    :param target:
    :param mask:
    :return:
    """
    total = mask.sum()
    crossEntropy = -torch.log(
        torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, total.item()


def train(input_variable, lengths, target_variable, mask, max_target_len,
          encoder, decoder, encoder_optimizer, decoder_optimizer,
          batch_size, clip, max_length=MAX_LENGTH):
    """

    :param input_variable:
    :param lengths:
    :param target_variable:
    :param mask:
    :param max_target_len:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param batch_size:
    :param clip:
    :param max_length:
    :return:
    """
    teacher_forcing_ratio = 0.1
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = False
    if random.random() < teacher_forcing_ratio:
        use_teacher_forcing = True

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t],
                                            mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor(
                [[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t],
                                            mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
               decoder_optimizer, embedding, encoder_n_layers,
               decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename):
    """

    :param model_name:
    :param voc:
    :param pairs:
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param embedding:
    :param encoder_n_layers:
    :param decoder_n_layers:
    :param save_dir:
    :param n_iteration:
    :param batch_size:
    :param print_every:
    :param save_every:
    :param clip:
    :param corpus_name:
    :param loadFilename:
    """
    # Load batches for each iteration
    training_batches = [
        batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = \
            training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask,
                     max_target_len, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print(
                "Iteration: {}; Percent complete: {:.1f}%; Average loss: {"
                ":.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}_{}'.format(encoder_n_layers,
                                                       decoder_n_layers,
                                                       # max_target_len))
                                                       hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en'       : encoder.state_dict(),
                'de'       : decoder.state_dict(),
                'en_opt'   : encoder_optimizer.state_dict(),
                'de_opt'   : decoder_optimizer.state_dict(),
                'loss'     : loss,
                'voc_dict' : voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory,
                            '{}_{}.tar'.format(iteration, 'checkpoint')))


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    """

    :param encoder:
    :param decoder:
    :param searcher:
    :param voc:
    :param sentence:
    :param max_length:
    :return:
    """
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    """

    :param encoder:
    :param decoder:
    :param searcher:
    :param voc:
    """
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc,
                                    input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if
                               not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def createEmbeddingLayer(weights_matrix, non_trainable=False):
    """

    :param weights_matrix:
    :param non_trainable:
    :return:
    """
    number_of_embeddings = len(weights_matrix)
    embedding_dimension = len(weights_matrix[0])
    print('Number of embeddings: ', number_of_embeddings)
    print('Dimension of embeddings: ', embedding_dimension)
    embedding_layer = nn.Embedding(number_of_embeddings, embedding_dimension)

    # embedding_layer.load_state_dict({'weight': weights_matrix},
    # strict=False)
    if non_trainable:
        embedding_layer.weight.requires_grad = False

    return embedding_layer, number_of_embeddings, embedding_dimension


# printLines(os.path.join(corpus, "AGREEMENT_BOT.txt"))

# Define path to new file
datafile = os.path.join(corpus, "output", "formatted_dialog_lines.txt")

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
# MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
# MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID",
#                               "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
# for file in os.listdir(corpus):
#     print('FileName: ', file)
#     print('Corpus: ', corpus)
#     printLines(os.path.join(corpus, file), 10)
# lines += loadLines(os.path.join(corpus, file))

# print("\nLoadingConversations...")
# conversations = loadConversations(
#     os.path.join(corpus, "movie_conversations.txt"), lines,
#     MOVIE_CONVERSATIONS_FIELDS)

# Write tsv file
print("\nWriting newly formatted file...")

with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for file in os.listdir(corpus):
        if not os.path.isdir(file) and file.find('.txt') > 0:
            print("FILE NAME: ", file)
            for pair in loadLines(os.path.join(corpus, file)):
                writer.writerow(pair)

# Print some ot the lines
# print("\nSome lines of the file\n")
# printLines(datafile)

# Load/Assemble voc and pairs
# Create string for datetime
now = datetime.datetime.now()
datetime_str = now.strftime("%Y%m%d%H%M%S")
# save_dir = os.path.join("data", "save", "300dEmbeddings")
save_dir = os.path.join("data", "save", datetime_str)
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate
print("\npairs:\n")
for pair in pairs[:10]:
    print(pair)

# Check for spelling errors. This should be done before removing rare words,
# as misspelled words should be rare by definition.
# pairs = spell_checker(pairs)
# TODO: Some issue with the output, which is not exactly as it is expected by
#  the next step. Should run it again without the correction part to fix it.

pairs = trimRareWords(voc, pairs, MIN_COUNT)

# Example for validation
small_batch_size = 10
batches = batch2TrainData(voc, [random.choice(pairs) for _ in
                                range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:\n", input_variable)
print("lengths:\n", lengths)
print("target_variable:\n", target_variable)
print("mask:\n", mask)
print("max_target_len:\n", max_target_len)

# Configure models
model_name = 'cb_model'
attention_model = 'dot'
# attention_model = 'general'
# attention_model = 'concat'
hidden_size = 500  # default 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.2  # default 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers,
#                            decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

# vectors = bcolz.open(f'{glove_path}/6B.300.dat')[:]
# words = pickle.load(open(f'{glove_path}/6B.300_words.pkl', 'rb'))
# word2idx = pickle.load(open(f'{glove_path}/6B.300_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}
print('GLOVE: ')
# i = 0
# for item in word2idx:
#     i += 1
#     print(item.value())
#     if i is 100:
#         break

print('Building encoder and decoder ...')
# Load glove embeddings
print("Number of words in vocabulary: ", voc.num_words)
matrix_length = voc.num_words
weights_matrix = np.zeros((voc.num_words, EMBEDDING_DIMENSION))
# print('Length of weights_matrix: ', len(weights_matrix))
words_found = 0
# print("voc: ", voc.index2word.values())
# print("voc index2word.values() size: ", len(voc.index2word.values()))
# print('glove.values() size: ', len(glove.values()))
for i, word in enumerate(voc.index2word.values()):
    # print('i: ', i)
    try:
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6,
                                             size=EMBEDDING_DIMENSION)

embedding_layer, number_embeddings, embeddings_dimensions = \
    createEmbeddingLayer(weights_matrix)

# Initialize word embeddings
# embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding_layer.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding_layer, encoder_n_layers, dropout)
decoder = LuongAttentionDecoderRNN(attention_model, embedding_layer,
                                   hidden_size,
                                   voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Configure training/optimization
clip = 30.0
teacher_forcing_ratio = 0.8  # default: 1
learning_rate = 0.00008
decoder_learning_ratio = 5.0  # default: 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate *
                                                        decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer,
           decoder_optimizer, embedding_layer, encoder_n_layers,
           decoder_n_layers, save_dir, n_iteration, batch_size, print_every,
           save_every, clip, corpus_name, loadFilename)

encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder, SOS_token)

evaluateInput(encoder, decoder, searcher, voc)
