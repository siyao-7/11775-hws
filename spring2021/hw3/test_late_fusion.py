#!/bin/python

import argparse
import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import time
import numpy as np

# Apply the SVM model to the testing videos;
# Output the prediction class for each video


parser = argparse.ArgumentParser()
parser.add_argument("model_file")
parser.add_argument("list_videos")
parser.add_argument("output_file")

parser.add_argument("--feat_dir", default="avgpool18/")
parser.add_argument("--feat_dir2", default="i3d/")
parser.add_argument("--feat_dim", type=int, default=256)
parser.add_argument("--feat_dim2", type=int, default=1024)
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--feat_appendix2", default="-rgb.npz")

if __name__ == '__main__':

  args = parser.parse_args()
  start = time.time()
  # 1. load svm model
  svm1 = pickle.load(open(args.model_file+".sound", "rb"))
  svm2 = pickle.load(open(args.model_file+".image", "rb"))
  svm3 = pickle.load(open(args.model_file+".fuse", "rb"))

  # 2. Create array containing features of each sample
  fread = open(args.list_videos, "r")
  feat_list = []
  feat_list2 = []
  video_ids = []
  for line in fread.readlines():
    # HW00006228
    video_id = os.path.splitext(line.strip())[0]
    video_ids.append(video_id)
    # pdb.set_trace()
    #feat_filepath = os.path.join(args.feat_dir, video_id + ".kmeans.csv")
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    if not os.path.exists(feat_filepath):
      feat_list.append(np.zeros(args.feat_dim))
    else:
      feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype='float'))
    
    feat_filepath2 = os.path.join(args.feat_dir2, video_id + args.feat_appendix2)
    if not os.path.exists(feat_filepath2):
      feat_list2.append(np.zeros(args.feat_dim2))
    else:
      tmp = np.load(feat_filepath2)
      tmp_feat = np.mean(np.squeeze(tmp.f.feature),axis=0)
      if tmp_feat.shape != (1024,):
          feat_list2.append(np.zeros(args.feat_dim2))
      else:
          feat_list2.append(tmp_feat)

  X1 = np.array(feat_list)
  X2 = np.array(feat_list2)
  X3 = np.concatenate((svm1.decision_function(X1)*0.6,svm2.decision_function(X2)),axis=1)

  # 3. Get scores with trained svm model
  # (num_samples, num_class)
  scoress = svm3.decision_function(X3)

  # 4. save the argmax decisions for submission
  with open(args.output_file, "w") as f:
    f.writelines("Id,Category\n")
    for i, scores in enumerate(scoress):
      predicted_class = np.argmax(scores)
      f.writelines("%s,%d\n" % (video_ids[i], predicted_class))
  end = time.time()
  print("Time for computation: ", (end - start))