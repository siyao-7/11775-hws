# Instructions for hw2

In this homework we will perform a video classification task using visual features.

## Video Data and Labels

Please download the videos from [this link](https://drive.google.com/file/d/1Oyzv7eC0QDrg0vX3AdSXYzdsFpIsdzT-/view?usp=sharing) or [this link](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/hw_11775_videos.zip). Then download the labels from [here](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/labels_v2.zip). Extract these zip files to get videos and their labels by:
```
$ unzip hw_11775_videos.zip
$ unzip labels_v2.zip
```

## Overirew
A video-based MED system is composed of mainly three parts: 1) Video pre-processing, 2) video feature extraction, and 3) classification. You will mainly focus on learning how to use two-types of visual features in this homework. You may simply reuse the classifcation codebase you finished in HW1. 

## Video Pre-Processing
There are many types of video features. For simplicity, we ask you to extract and aggregate features of RGB frames from videos. For pre-processing, we fist use ffmpeg you learned in HW1 to extract the video frames:

```
for video in videos/*.mp4
do
    base=$(basename -- "$video")
    base="${base%.*}"
    mkdir -p rgb/$base
    ffmpeg -i "$video"  rgb/"$base"/frame_%05d.jpg -hide_banner
done
```

## Extract SURF feature
For the hand-crafted visual feature. We ask you to Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...). We provide a simple template for you to work on SURF:

```
python surf_feat_extraction.py -i list/all.video config.yaml
```

Please check surf_feat_extraction.py for details.


## Extract CNN features
In HW2 we ask you to extract CNN-based feature for each frame. We provide an example using ResNet, a type of CNN model that has been widely used in many computer vision tasks. To extract the feature of an image. You may use:
```
model = Get_CNN(cuda=False, model_name='resnet34', layer='avgpool', layer_output_size=512)
emb = model.get_emb(Image.open('test.jpg'))
```
Please check extract_resnet.py for the details. After extracting CNN-based feature, you can choose to apply and build bag-of-words feature as in HW1 or simply use the original feature extracted. You are alos encouraged to try some more advanced model such as DenseNet and RexNeXt. 


## Training and Testing Classifiers
As you already have the code to train and test SVM and MLP.
**You can reuse your code from HW1 for bag-of-words and SVM/MLP training.**


## Potential Improvement
Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:
+ Use more advanced visual models/features such as DenseNet and ResNeXt.
+ End-to-end training of these models instead of just extracting features.
+ Use video models/features such as I3D that takes video clips as its input insteadt of frames.
