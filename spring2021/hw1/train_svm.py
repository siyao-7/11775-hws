#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
import pickle

import sys
import pdb

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
  if len(sys.argv) != 6:
    print("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
    print("cat_name -- category")
    print("feat_dir -- dir of feature files")
    print("feat_dim -- dim of features")
    print("output_file -- path to save the svm model")
    print("list_videos -- file containing videos for training with event labels")
    exit(1)

  event_name = sys.argv[1]
  feat_dir = sys.argv[2]
  feat_dim = int(sys.argv[3])
  output_file = sys.argv[4]
  list_videos = sys.argv[5]

  # 1. read all features in one array.
  # Use list of videos in training set and read only those features.
  fread = open(list_videos, "r")
  feat_list = []
  # if event occured: 1 if not: 0; labels
  label_list = []
  # load video names and events in dict
  #df_videos_label = pd.read_csv(list_videos, delimiter=' ')
  df_videos_label = {}
  for line in open(list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    # pdb.set_trace()
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(feat_dir, video_id + ".kmeans.csv")
    # for videos with no audio, create feature and label
    if os.path.exists(feat_filepath):
      # pdb.set_trace()
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

      label_list.append(int(df_videos_label[video_id] == event_name))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # pass array for svm training
  # pdb.set_trace()
  clf = SVC(cache_size=2000)
  clf.fit(X, y)

  # save trained SVM in output_file
  pickle.dump(clf, open(output_file, 'wb'))
  print('SVM trained successfully for event %s!' % (event_name))
