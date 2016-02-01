#!/bin/python
import numpy
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import collections
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]
    file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model, "rb"))

    fread = open(file_list, "r")
    hist_path = "kmeans/all.mfcc.hist"
    fwrite = open(hist_path, "w")

    for line in fread.readlines():
        mfcc_path = "mfcc/" + line.replace('\n', '') + ".mfcc.csv"
        if os.path.exists(mfcc_path) is False:
            continue
        X = numpy.genfromtxt(mfcc_path, delimiter=";")
        labels = kmeans.predict(X)
        counter = collections.Counter(labels)
        vector = [counter[n] for n in xrange(cluster_num)]
        line = ';'.join([str(v) for v in vector])
        fwrite.write(line + '\n')
    fread.close()
    fwrite.close()

    print "K-means features generated successfully!"
