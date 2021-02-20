#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
import time
import collections
import csv
import argparse
from tqdm import tqdm
# Generate k-means features for videos; each video is represented by a single vector

parser = argparse.ArgumentParser()
parser.add_argument("kmeans_model")
parser.add_argument("cluster_num", type=int)
parser.add_argument("file_list")
parser.add_argument("--mfcc_path", default="mfcc")
parser.add_argument("--output_path", default="bof")

if __name__ == '__main__':
  args = parser.parse_args()


  # 1. load the kmeans model
  kmeans = pickle.load(open(args.kmeans_model, "rb"))

  # 2. iterate over each video and use kmeans.predict(mfcc_features_of_video) and save the results in this format video_name.kmeans
  start = time.time()
  fread = open(args.file_list, "r")
  for line in tqdm(fread.readlines()):
    mfcc_path = os.path.join(args.mfcc_path, line.strip() + ".mfcc.csv")
    bof_path = os.path.join(args.output_path, line.strip() + ".csv")

    if not os.path.exists(mfcc_path):
      continue

    array = numpy.genfromtxt(mfcc_path, delimiter=";")
    kmeans_result = kmeans.predict(array)

    dict_freq = collections.Counter(kmeans_result)
    # create dict containing 0 count for cluster number
    keys = numpy.arange(0, args.cluster_num, 1)
    values = numpy.zeros(args.cluster_num, dtype="float")
    dict2 = dict(zip(keys, values))
    dict2.update(dict_freq)
    list_freq = list(dict2.values())
    # normalize the frequency by dividing with frame number
    list_freq = numpy.array(list_freq) / array.shape[0]
    numpy.savetxt(bof_path, list_freq)

  end = time.time()
  print("K-means features generated successfully!")
  print("Time for computation: ", (end - start))
