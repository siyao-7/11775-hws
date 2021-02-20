#!/bin/python

import numpy as np
import os
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys

# Performs K-means clustering and save the model to a local file

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':

  args = parser.parse_args()

  # 1. read all features in one array.
  # Use list of videos in training set and read only those features.
  fread = open(args.list_videos, "r")
  feat_list = []
  # if event occured: 1 if not: 0; labels
  label_list = []
  # load video names and events in dict
  #df_videos_label = pd.read_csv(list_videos, delimiter=' ')
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    # pdb.set_trace()
    video_id = line.strip().split(",")[0]
    #feat_filepath = os.path.join(args.feat_dir, video_id + ".kmeans.csv")
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, create feature and label
    if os.path.exists(feat_filepath):
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)

  # pass array for svm training
  # pdb.set_trace()
  # one-versus-rest multiclass strategy
  clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam")
  clf.fit(X, y)

  # save trained SVM in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
