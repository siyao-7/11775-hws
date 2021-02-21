# coding=utf-8
# collect the soundnet features into csv (by global averaging)
import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("feature_path")
parser.add_argument("feature_name")
parser.add_argument("outpath")
parser.add_argument("--use_average", action="store_true")

if __name__ == "__main__":
  args = parser.parse_args()

  videos = glob(os.path.join(args.feature_path, "*"))

  if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)

  for video in tqdm(videos):
    # tf_fea%01d.npy
    feature_file = os.path.join(video, "%s.npy" % args.feature_name)
    video_id = video.strip("/").split("/")[-1]

    # (T, C)
    # T depends on the audio length
    # C is the number of filters for that CNN layer
    feature = np.load(feature_file)

    new_feature_file = os.path.join(args.outpath, "%s.csv" % video_id)

    if args.use_average:
      # global averaging
      feature = np.mean(feature, axis=0)  # (C,)

    with open(new_feature_file, "w") as f:
      for i in range(len(feature)):
        if args.use_average:
          feature_list = [str(feature[i])]
        else:
          feature_list = [str(o) for o in feature[i]]
        # semicolon separated, to be consistent with MFCC-BoF outputs
        f.writelines("%s\n" % (";".join(feature_list)))

