# Instructions for hw1

In this homework we will perform a video classification task using audio-only features.

## Data and Labels

Please download the videos from [this link](https://drive.google.com/file/d/1Oyzv7eC0QDrg0vX3AdSXYzdsFpIsdzT-/view?usp=sharing) or [this link](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/hw_11775_videos.zip). Then download the labels from [here](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/labels.zip).

## Step-by-step baseline instructions

For the baselines, we will provide code and instructions for two feature representations (MFCC-Bag-of-Features and [SoundNet-Global-Pool](https://arxiv.org/pdf/1610.09001.pdf)) and two classifiers (SVM and MLP). Assuming you are under Ubuntu 18.04 system and under this directory (11775-hws/spring2021/hw1/).

First, put the videos into `videos/` folder and the labels into `labels/` folder simply by:
```
$ unzip hw_11775_videos.zip
$ unzip labels.zip
```

### MFCC-Bag-Of-Features

Let's create the folders we need first:
```
$ mkdir audio/ mfcc/ bof/
```

1. Dependencies: FFMPEG, OpenSMILE, Python: sklearn, pandas

Download OpenSMILE 2.3.0 from [here](https://he.audeering.com/download/opensmile-2-3-0-tar-gz/) and then extract in this directory:
```
$ tar -zxvf opensmile-2.3.0.tar.gz
```

Install FFMPEG by:
```
$ sudo apt install ffmpeg
```

Install python dependencies by:
```
$ sudo pip install sklearn pandas tqdm
```

2. Get MFCCs

We will first extract wave files and then run OpenSMILE to get MFCCs into CSV files. We will directly run the binaries of OpenSMILE (no need to install):
```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav audio/${filename}.wav; ./opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C config/MFCC12_0_D_A.conf -I audio/${filename}.wav -O mfcc/${filename}.mfcc.csv;done
```

Note that some audio/mfcc files might be missing. This is due to the fact that some videos have no audio, which is common in real-world scenario. We'll learn to deal with that.

3. K-Means clustering

As taught in the class, we will use K-Means to get feature codebook from the MFCCs. Since there are too many feature lines, we will randomly select a subset for K-Means clustering by:
```
$ python2 select_frames.py labels/trainval.csv 0.2 selected.mfcc.csv --mfcc_path mfcc/
```

Now we train it by (50 clusters, this would take about 7-10 minutes):
```
$ python2 train_kmeans.py selected.mfcc.csv 50 kmeans.50.model
```

4. Feature extraction

Now we have the codebook, we will get bag-of-features (a.k.a. bag-of-words) using the codebook and the MFCCs. First, we need to get video names:
```
$ ls videos/|while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
```


Now we extract the feature representations for each video:
```
$ python2 get_bof.py kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

Now you can follow [here] to train SVM classifiers or [MLP]() ones.

### SoundNet-Global-Pool

Just as the MFCC-Bag-Of-Feature, we will also use the [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) model to extract a vector feature representation for each video. Since SoundNet is trained on a large dataset, this feature is usually better compared to MFCCs.

1. Install dependencies

We will use our fork of the SoundNet-Tensorflow repo:
```
$ git clone https://github.com/JunweiLiang/SoundNet-tensorflow
```

We will need librosa to read audio files:
```
$ sudo pip2 install llvmlite==0.31.0
$ sudo pip2 install librosa
```

And Tensorflow:
```
$ sudo pip2 install tensorflow-gpu==1.15.0
```

Download the pretrained SoundNet model:
```
$ cd SoundNet-tensorflow
$ mkdir models; cd models
$ wget https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/sound8.npy
$ cd ..
```

2. Extract features for each video

SoundNet has 8 convolutional layers. You can see full layer definition in the paper and in `models.py`. We will use conv4 [index: 14, dimension: 128] as an example and you can experiment with others.

Suppose you are under the `SoundNet-tensorflow` directory. First get the audio file into a list:
```
$ ls ../audio/|while read line;do echo "../audio/${line}";done > data.lst
```

Extract the feature by (this takes a couple of minutes):
```
$ python2 -W ignore extract_feat.py -m 14 -x 15 -s -p extract -t data.lst -o features/
```

Since the audio is of different length, the features will have variable rows. We get a fixed-length vector representation for each video by global averaging:
```
$ python2 ../collect_soundnet.py features/ tf_fea14 ../soundnet_fea14_avg --use_average
```

### SVM classifier

From the previous sections, we have extracted two fixed-length vector feature representations for each video. We will use them separately to train classifiers.

Suppose you are under `hw1` directory. Train SVM by:
```
$ mkdir models/
$ python2 train_svm_multiclass.py bof/ 50 labels/trainval.csv models/mfcc-50.svm.multiclass.model
```

Run SVM on the test set:
```
$ python2 test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 labels/test_for_student.label mfcc-50.svm.multiclass.csv
```

### MLP classifier

Suppose you are under `hw1` directory. Train MLP by:
```
$ python2 train_mlp.py bof/ 50 labels/trainval.csv models/mfcc-50.mlp.model
```

Test:
```
$ python2 test_mlp.py models/mfcc-200.mlp.model bof 50 labels/test_for_student.label mfcc-200.mlp.csv
```

### Submission

You can then submit the test outputs to the leaderboard:
```
https://www.kaggle.com/c/11775-hw1
```
We use accuracy as our metric.

### Things to try to improve your model performance

Now here comes the fun part. You can start experimenting with the code and exploring how you can improve your model performance. Some hints:

+ Split `trainval.csv` into `train.csv` and `val.csv` to validate your model variants. This is important since the leaderboard limits the number of times you can submit, which means you cannot test most of your experiments on the official test set.
+ Try different number of K-Means clusters
+ Try different layers of SoundNet
+ Try different classifiers (different SVM kernels, different MLP hidden sizes, etc.). Please refer to [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier) documentation.

