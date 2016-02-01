#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "vocab_file -- path to the vocabulary file"
        print "file_list -- the list of videos"
        exit(1)

    vocab_file = sys.argv[1]
    file_list = sys.argv[2]

    # create reverted index for vocab
    vocab = numpy.genfromtxt(vocab_file, dtype=str)
    vocab_index = {}
    for i, v in enumerate(vocab):
        vocab_index[v] = i

    fread = open(file_list, "r")
    hist_path = "asrfeat/all.asr.hist"
    fwrite = open(hist_path, "w")

    for line in fread.readlines():
        asr_path = "asr/" + line.replace('\n', '') + ".ctm"
        if os.path.exists(asr_path) is False:
            continue
        X = numpy.genfromtxt(asr_path, dtype=None, delimiter=" ")
        vector = [0]*len(vocab)
        for x in X:
            word = x[4]
            if word not in vocab_index:
                continue
            vector[vocab_index[word]] += 1
            norm = numpy.linalg.norm(vector)
            if norm > 0:
                vector = vector/norm
        line = ';'.join([str(v) for v in vector])
        fwrite.write(line + '\n')
    fwrite.close()

    print "ASR features generated successfully!"
