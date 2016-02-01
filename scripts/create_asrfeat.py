#!/bin/python
import numpy
import os
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

    for line in fread.readlines():
        video_id = line.replace('\n', '')
        asr_path = "asr/" + video_id + ".ctm"
        if os.path.exists(asr_path) is False:
            continue
        vector = [0]*len(vocab)
        fread_asr = open(asr_path, "r")
        for line_asr in fread_asr.readlines():
            tokens = line_asr.strip().split(' ')
            word = tokens[4]
            if word not in vocab_index:
                continue
            vector[vocab_index[word]] += 1
        fread_asr.close()
        s = sum(vector)
        if s > 0:
            vector = vector/s
        line = ';'.join([str(v) for v in vector])
        feat_path = "asrfeat/"+video_id+".feat"
        fwrite = open(feat_path, "w")
        fwrite.write(line + '\n')
        fwrite.close()
    fread.close()

    print "ASR features generated successfully!"
