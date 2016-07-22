import numpy as np
from collections import defaultdict
import seaborn as sb
from scipy.spatial import distance
import sys

# TODO: use the following training data (context|target) and see if 100/500 iterations
# results in the visualization similar to https://ronxin.github.io/wevi/

# eat|apple,eat|orange,eat|rice,drink|juice,drink|milk,drink|water,orange|juice,apple|juice,rice|milk,milk|drink,water|drink,juice|drink
# the|dog,dog|saw,saw|a,a|cat,the|dog, dog|chased,chased|the,the|cat,the|cat,cat|climbed,climbed|a,a|tree
#sentences = ["eat apple","eat orange","eat rice","drink juice","drink milk","drink water","orange juice","apple juice","rice milk","milk drink","water drink","juice drink"]
sentences = ['the dog saw a cat', 'the dog chased the cat', 'the cat climbed a tree']
# sentences = ['the king loves the queen', 'the queen loves the king',
#              'the dwarf hates the king', 'the queen hates the dwarf',
#              'the dwarf poisons the king', 'the dwarf poisons the queen']


def Vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary

def docs2bow(sentences, dictionary):
    """Transforms a list of strings into a list of lists where
    each unique item is converted into a unique integer."""
    for sentence in sentences:
        yield [dictionary[word] for word in sentence.split()]

vocabulary = Vocabulary()
sentences_bow = list(docs2bow(sentences, vocabulary))
sentences_bow

V, N = len(vocabulary), 5
#V, N = len(vocabulary), 2 # for easy visualization
WI = (np.random.random((V, N)) - 0.5) / N
WO = (np.random.random((N, V)) - 0.5) / V

# target_word = 'king'
# input_word = 'queen'
learning_rate = 0.2

# Once the error is known, the weights in the matrices WO and WI can be updated using backpropagation.
# Thus, the training can proceed by presenting different context-target words pair from the corpus.
# In essence, this is how Word2vec learns relationships between words and in the process develops vector
# representations for words - called word embeddings, the final WI?! - in the corpus.
# TODO: verify the weight updates' correctedness with the word2vec-explained.pdf
# TODO: understand word2vec open source code; tensorflow word2vec_basic.py
# TODO: visualize the word embeddings - WI?! - after multiple (how many?) calls to updateWeights.

# input_word is the same as context_word
def updateWeights(input_word, target_word):
    ys = np.array([])
    errors = {}
    for word in vocabulary:
        top = np.exp(-np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]]))
        top = sys.float_info.max if top == float('inf') else top
        y = top / sum(sys.float_info.max if np.exp(-np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]])) == float('inf') else np.exp(-np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]])) for w in vocabulary)
#        y = top / sum(np.exp(np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]])) for w in vocabulary)
        print ">>>", top, y
        ys = np.append(ys, y)
        t = 1 if word == target_word else 0
        error = t - y
        errors[word] = error
        print word, y, error

    for word in vocabulary:
        WO.T[vocabulary[word]] = (WO.T[vocabulary[word]] - learning_rate * errors[word] * WI[vocabulary[input_word]])
        print WO

    #print ys, sum(ys)
    # TODO: this line seems problematic! figure out why? (after 20 or more training(), some
    # WI and WO values are way bigger than 1)
    WI[vocabulary[input_word]] = WI[vocabulary[input_word]] - learning_rate * WO.sum(1)


    # for word in vocabulary:
    #     p = (np.exp(np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]])) /
    #         sum(np.exp(np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]])) for w in vocabulary))
    #     print word, p


# Suppose we want the network to learn relationship between the words “cat” and “climbed”. That is, the network
# should show a high probability for “climbed” when “cat” is inputted to the network. In word embedding
# terminology, the word “cat” is referred as the context word and the word “climbed” is referred as the
# target word.
#updateWeights("climbed", "cat")
#updateWeights("chased", "dog")
def training():
    for sentence in sentences:
        first = sentence.split()[:-1]
        second = sentence.split()[1:]
        for (context_word, target_word) in zip(first, second):
            print context_word, target_word
            updateWeights(context_word, target_word)



# another way (non-visulization test) to calculate the similarity of two words
# inspired by https://ronxin.github.io/wevi/

words = [word for (word, index) in vocabulary.iteritems()]
word_test = "chased"
for w in words:
    print word_test, ":", w, "=", distance.euclidean(WO.T[vocabulary[word_test]], WO.T[vocabulary[w]])

for word_input in words:
    for word_output in words:
        print word_input, ":", word_output, "=", distance.euclidean(WI[vocabulary[word_input]], WO.T[vocabulary[word_output]])


# visualization to prove it makes sense (if the size of hidden layer is 2)
embeddings = WI # final result of word vectors
words = [word for (word, index) in vocabulary.iteritems()]
sb.plt.plot(embeddings[:,0], embeddings[:,1], 'o')
x = embeddings[:,0]
y = embeddings[:,1]
sb.plt.xlim(min(x) - 0.2 * abs(min(x)), max(x) + 0.2 * abs(max(x)))
sb.plt.ylim(min(y) - 0.2 * abs(min(y)), max(y) + 0.2 * abs(max(y)))
for word, x, y in zip(words, embeddings[:,0], embeddings[:,1]):
    sb.plt.annotate(word, (x, y), size=16)
sb.plt.show()

# for each input word, calcluate its distance to each output word
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.sum(np.square(v1))) * np.sqrt(np.sum(np.square(v2))))

# after calling training() once:
# saw : a = 0.147484084936
# saw : tree = 0.156660341506
# saw : climbed = 0.128953896198
# saw : chased = 0.153088889615
# saw : dog = 0.177312318116
# saw : cat = 0.0962355118321
# saw : the = 0.114696079368
# saw : saw = 0.0
#
# after calling training() 5 more times:
# saw : a = 0.319842439962
# saw : tree = 0.277418586527
# saw : climbed = 0.37132067148
# saw : chased = 0.178100571474
# saw : dog = 1.16069120022
# saw : cat = 1.72759543586
# saw : the = 0.336707314938
# saw : saw = 0.0
#
# after 5 more times - "saw" and "chased" are closest!
# saw : a = 0.629148878277
# saw : tree = 0.743367324074
# saw : climbed = 12.3264739598
# saw : chased = 0.284198210389
# saw : dog = 5.41812117961
# saw : cat = 7.61815835293
# saw : the = 1.3702822688
# saw : saw = 0.0

# TODO: figure out why my code has nan in weights after about 20-30 training
# DONE - bug in updateWeights: WO shouldn't be changed while looping over vocabulary

# TODO: fully understand wevi JS code and word2vec open source C code
# climbed 1.0
# chased 9.17475883287e-170
# dog 2.4600780222e-208
# cat 1.69146253184e-223
# the 4.51222594411e-162
# saw 5.59428539901e-169

for word in vocabulary:
    p_word_queen = (np.exp(-np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]])) /
        sum(np.exp(-np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]])) for w in vocabulary))
    print p_word_queen
