# HW1

## Feature extraction
SURF: use the same step as sample code https://github.com/11775website/11775-hws/tree/master/spring2021/hw2 <br />
CNN: python cnn_feat_extraction.py videos/ cnn_feat/ --use_gpu
I3D: follow instructions in https://github.com/Finspire13/pytorch-i3d-feature-extraction, put the result under folder i3d/

## For testing
Generate csv for testing SURF, CNN and I3D with SVM classifier
```
SURF
$ python3 test_svm_multiclass.py models/surf.50.svm.model surf_bof/ 50 labels/test_for_student.label surf.csv
CNN
$ python3 test_svm_multiclass.py models/cnn.512.svm.model cnn_feat/ 512 labels/test_for_student.label cnn.csv
I3D
$ python3 test_svm_multiclass_i3d.py models/i3d.svm.model i3d/ 1024 labels/test_for_student.label best.csv
```

## For training
```
SURF
$ python3 train_svm_multiclass.py surf_bof/ 50 labels/train.csv labels/val.csv models/surf.50.svm.model
CNN
$ python3 train_svm_multiclass.py cnn_feat/ 512 labels/train.csv labels/val.csv models/cnn.512.svm.model
I3D
$ python3 train_svm_multiclass_i3d.py i3d/ 1024 labels/train.csv labels/val.csv models/i3d.svm.model
```


