#!/bin/python

import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import pdb

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
  if len(sys.argv) != 6:
    print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
    print("model_file -- path of the trained svm file")
    print("feat_dir -- dir of feature files")
    print("feat_dim -- dim of features; provided just for debugging")
    print("output_file -- path to save the prediction score")
    print("list_videos -- file containing videos for testing with event labels")
    exit(1)

  model_file = sys.argv[1]
  feat_dir = sys.argv[2]
  feat_dim = int(sys.argv[3])
  output_file = sys.argv[4]
  list_videos = sys.argv[5]

  # 1. load svm model
  svm = pickle.load(open(model_file, "rb"))

  # 2. Create array containing features of each sample
  fread = open(list_videos, "r")
  feat_list = []

  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    # pdb.set_trace()
    feat_filepath = os.path.join(feat_dir, video_id + ".kmeans.csv")
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

  X = np.array(feat_list)

  # 3. Get scores with trained svm model
  scores = svm.decision_function(X)

  # 4. Save scores
  np.savetxt(output_file, scores)
