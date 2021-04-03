#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
import sklearn.metrics as m
import pickle
import argparse
import sys
import pdb

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("output_file")
parser.add_argument("--feat_dir", default="avgpool18/")
parser.add_argument("--feat_dir2", default="i3d/")
parser.add_argument("--feat_dim", type=int, default=256)
parser.add_argument("--feat_dim2", type=int, default=1024)
parser.add_argument("--list_videos", default="labels/train.csv")
parser.add_argument("--val_videos", default="labels/val.csv")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--feat_appendix2", default="-rgb.npz")

if __name__ == '__main__':
  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  feat_list2 = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    feat_filepath2 = os.path.join(args.feat_dir2, video_id + args.feat_appendix2)
    
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath) and os.path.exists(feat_filepath2):
      tmp = np.load(feat_filepath2)
      tmp_feat = np.mean(np.squeeze(tmp.f.feature),axis=0)
      if tmp_feat.shape != (1024,):
          continue
      feat_list2.append(tmp_feat)
    
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))
        
  fread.close()

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X1 = np.array(feat_list)
  X2 = np.array(feat_list2)
    
  print("late fusion")
  
  # pass array for svm training
  clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", tol=1e-3, C=20.0)
  clf.fit(X1, y)

  pickle.dump(clf, open(args.output_file+".sound", 'wb'))
  print('SVM trained successfully on sound features')

  clf2 = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", tol=1e-3, C=15.0)
  clf2.fit(X2, y)

  pickle.dump(clf2, open(args.output_file+".image", 'wb'))
  print('SVM trained successfully on image features')

#   predicted_class1 = np.argmax(clf.decision_function(X1), 1)
#   predicted_class1 = predicted_class1.reshape(predicted_class1.shape[0],1)
#   predicted_class2 = np.argmax(clf2.decision_function(X2), 1)
#   predicted_class2 = predicted_class2.reshape(predicted_class2.shape[0],1)
#   X3 = np.concatenate((predicted_class1,predicted_class2),axis=1)
  X3 = np.concatenate((clf.decision_function(X1)*0.6,clf2.decision_function(X2)),axis=1)
  clf3 = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", tol=1e-3, C=15.0)
  clf3.fit(X3, y)

  pickle.dump(clf3, open(args.output_file+".fuse", 'wb'))
  print('SVM trained successfully on late fusion')
    

  # val
  fread = open(args.val_videos, "r")
  feat_list = []
  feat_list2 = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.val_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    feat_filepath2 = os.path.join(args.feat_dir2, video_id + args.feat_appendix2)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath) and os.path.exists(feat_filepath2):
      tmp = np.load(feat_filepath2)
      tmp_feat = np.mean(np.squeeze(tmp.f.feature),axis=0)
      if tmp_feat.shape != (1024,):
          continue
      feat_list2.append(tmp_feat)
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
      label_list.append(int(df_videos_label[video_id]))
  fread.close()

  X1 = np.array(feat_list)
  X2 = np.array(feat_list2)

#   predicted_class1 = np.argmax(clf.decision_function(X1), 1)
#   predicted_class1 = predicted_class1.reshape(predicted_class1.shape[0],1)
#   predicted_class2 = np.argmax(clf2.decision_function(X2), 1)
#   predicted_class2 = predicted_class2.reshape(predicted_class2.shape[0],1)
#   X3 = np.concatenate((predicted_class1,predicted_class2),axis=1)
  X3 = np.concatenate((clf.decision_function(X1)*0.6,clf2.decision_function(X2)),axis=1)

  scoress = clf3.decision_function(X3)
  predicted_class = np.argmax(scoress, 1)
  acc = m.accuracy_score(np.array(label_list),predicted_class)
  print("dev acc:", acc)
  print("confusion matrix")
  print(m.confusion_matrix(np.array(label_list),predicted_class))
