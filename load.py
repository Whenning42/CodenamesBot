import numpy as np

FILE = "glove.6B.300d.txt"
LIST = "wordlist.txt"

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

words = []
with open(LIST) as wordlist_file:
    for w in wordlist_file:
        words.append(w.strip())

glove = loadGloveModel(FILE)
while True:
    prompt = input()
    print("Distances for ", prompt)
    for word in [prompt, *words]:
        print(word, np.dot(glove[prompt], glove[word]))
