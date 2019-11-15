import sys
import re

sys.path.append('/home/ooo/Documents/CS236/text-to-image-gen-model')

from transformers import *

from utils.pixelcnnpp_utils import *
import pdb
from torch.nn.utils import weight_norm as wn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import torchtext.vocab as vocab


def bert_encoder():
    return BERTEncoder()


def class_embedding(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


def unconditional(n_classes, embedding_dim):
    return nn.Embedding(n_classes, embedding_dim)


def glove_encoder():
    return GloveEncoder()


class Embedder(nn.Module):
    def __init__(self, embed_size):
        super(Embedder, self).__init__()
        self.embed_size = embed_size

    def forward(self, class_labels, captions):
        raise NotImplementedError


class BERTEncoder(Embedder):
    '''
    pretrained model used to embed text to a 768 dimensional vector
    '''

    def __init__(self):
        super(BERTEncoder, self).__init__(embed_size=768)
        self.pretrained_weights = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.model = BertModel.from_pretrained(self.pretrained_weights)
        self.max_len = 50

    def tokenize(self, text_batch):
        text_token_ids = [
            torch.tensor(self.tokenizer.encode(string_, add_special_tokens=False, max_length=self.max_len)) for
            string_ in text_batch]
        padded_input = pad_sequence(text_token_ids, batch_first=True, padding_value=0)
        return padded_input

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''

        padded_input = self.tokenize(captions)
        device = list(self.parameters())[0].device
        padded_input = padded_input.to(device)
        # takes the mean of the last hidden states computed by the pre-trained BERT encoder and return it
        return self.model(padded_input)[0].mean(dim=1)


class OneHotClassEmbedding(Embedder):

    def __init__(self, num_classes):
        super(OneHotClassEmbedding, self).__init__(embed_size=num_classes)
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.eye(self.num_classes))

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        return self.weights[class_labels]


class UnconditionalClassEmbedding(Embedder):
    def __init__(self):
        super(UnconditionalClassEmbedding, self).__init__(embed_size=1)

    def forward(self, class_labels, captions):
        '''
        :param class_ids : torch.LongTensor, class ids
        :param list text_batch: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        zero = torch.zeros(class_labels.size(0), 1).to(class_labels.device)
        return zero


class GloveEncoder(Embedder):
    '''
    glove to embed text
    '''

    def __init__(self):
        self.embedding_size = 300
        super(GloveEncoder, self).__init__(embed_size=self.embedding_size)
        self.glove = vocab.GloVe(name='6B', dim=self.embedding_size)
        print('Glove embedding size: ' + str(self.embedding_size))
        print('Loaded {} words'.format(len(self.glove.itos)))

        self.labelFileName = 'datasets/ImageNet32/map_clsloc.txt'
        self.embeddingMap = dict()
        self.buildEmbeddingMap()

    def isKnown(self, word):
        if word in self.glove.stoi:
            return True
        return False

    def word2id(self, word):
        # unknown word
        wordIdx = 1
        if word in self.glove.stoi:
            wordIdx = self.glove.stoi[word]
        return wordIdx

    def buildEmbeddingMap(self):
        labelFile = open(self.labelFileName, 'r')

        n_total = 0
        n_unknown = 0
        for line in labelFile:
            res = re.match('^(n\d+) (\d+) (\S+)', line)
            label = res.group(3)
            wList = re.split('_|-', label)
            cnt = 0
            avgTensor = torch.zeros([self.embedding_size])
            for word in wList:
                word = word.lower()
                word = word.replace('\'s', '')
                if self.isKnown(word):
                    wordIdx = self.word2id(word)
                    wordTensor = self.glove.vectors[wordIdx]
                    avgTensor += wordTensor
                    cnt += 1
                if cnt == 0:
                    n_unknown += 1
                    avgTensor = self.glove.vectors[1] # unknown label
                    #print(label)
            if cnt > 0:
                avgTensor /= cnt
            caption = label.replace("_", " ")
            self.embeddingMap[caption] = avgTensor
            n_total += 1

        labelFile.close()

        print('Total labels = ' + str(n_total))
        print('GloVe unknown labels = ' + str(n_unknown))

    def forward(self, class_labels, captions):
        '''
        :param class_labels : torch.LongTensor, class ids
        :param list captions: list of strings, sentences to embed
        :return: torch.tensor embeddings: embeddings of shape (batch_size,embed_size=768)
        '''
        device = class_labels.device
        #tList = [self.glove.vectors[self.word2id(word)] for word in captions]
        tList = [self.embeddingMap[caption] for caption in captions]
        t = torch.stack(tList)
        return t.to(device)

if __name__ == "__main__":
    print("Test GloVe on imagenet labels")

    embedder = GloveEncoder()

    n_total = 0
    n_unknown = 0
    labelFileName = 'datasets/ImageNet32/map_clsloc.txt'
    labelFile = open(labelFileName, 'r')

    for line in labelFile:
        res = re.match('^(n\d+) (\d+) (\S+)', line)
        label = res.group(3)
        label = label.replace("_", " ")
        #print(label)
        if not embedder.isKnown(label):
            n_unknown += 1
        n_total += 1

    labelFile.close()

    print('Total labels = ' + str(n_total))
    print('Unknown labels = ' + str(n_unknown))

#    print(embedder.forward([1], 'hello'))
