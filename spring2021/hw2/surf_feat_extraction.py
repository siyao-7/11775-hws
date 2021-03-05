#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import pickle
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("video_path")
parser.add_argument("output_path")
parser.add_argument("--feat_fps", type=float, default=1.5,
                    help="frame rate for getting features")
parser.add_argument("--hessian_threshold", type=float, default=400,
                    help="hyperparam for surf feature")

def get_surf_features_from_video(surf_detector,
                                 video_filepath,
                                 surf_feat_filepath,
                                 keyframe_interval):
  "Receives filename of downsampled video and of output path for features."
  " Extracts features in the given keyframe_interval. "
  " Saves features in pickled file."

  surf_feat = []

  for keyframe in get_keyframes(video_filepath, keyframe_interval):
    # key_points: the location of the detected key points of current frame
    # feat: the corresponding surf features around the key points
    key_points, feat = surf_detector.detectAndCompute(keyframe, None)
    # (440, 64) [-0.00128541  0.00350391  0.0021598   0.00519999  0.00177394 -0.00074453
    # 0.0114676   0.00843902  0.01255265 -0.01106742] ../videos/HW00002470.mp4
    # (num_keypoints, 64)
    #print(feat.shape, feat[0, :10], video_filepath)
    #sys.exit()
    surf_feat.append(feat)

  pickle.dump(surf_feat, open(surf_feat_filepath, 'wb'))
  # to open this file use:
  # pickle.load(open(surf_feat_video_filename, 'rb'), encoding='latin1')


def get_keyframes(video_filepath, keyframe_interval):
  """Generator function which returns the next keyframe.
  Args:
      video_filepath (string):
      keyframe_interval (int):
  Returns:
      frame (np.array):
  """

  video_cap = cv2.VideoCapture(video_filepath)

  video_cap.release()

if __name__ == '__main__':
  args = parser.parse_args()

  # Create SURF object
  # 1. need to reinstall opencv 4.5 from the following url
  # https://github.com/opencv/opencv-python/issues/126#issuecomment-621583923
  # 2. or downgrade:
  # $ sudo pip install opencv-contrib-python==3.4.2.16
  surf = cv2.xfeatures2d.SURF_create(hessianThreshold=args.hessian_threshold)

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  video_fps = 30.0  # assuming the video frame rate
  # extract feature from every k frame
  frame_interval = int(video_fps / args.feat_fps)

  # Loop over all videos (training, val, testing)
  video_files = glob(os.path.join(args.video_path, "*.mp4"))

  for video_file in tqdm(video_files):
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    surf_feat_outfile = os.path.join(args.output_path, video_name + ".p")

    # Get SURF features for one video
    get_surf_features_from_video(surf, video_file,
                                 surf_feat_outfile, frame_interval)
