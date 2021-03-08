# HW1

## Feature extraction
MFCC: use the same step as sample code https://github.com/11775website/11775-hws/tree/master/spring2021/hw1
SoundNet: follow the step in https://github.com/JunweiLiang/SoundNet-tensorflow to generate soundnet features. Store the results under /hw1/soundnet, and run python avgpool.py. The final result will be located in /hw1/avgpool18/

## For testing
Generate csv for testing mfcc and SoundNet with SVM and MLP classifier
```
MFCC with SVM
$ python3 test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 labels/test_for_student.label mfcc-50.svm.multiclass.csv
MFCC with MLP
$ python3 test_mlp.py models/mfcc-50.mlp.model bof 50 labels/test_for_student.label mfcc-50.mlp.csv
SoundNet with SVM
$ python3 test_svm_multiclass.py models/soundnet-avgpool18.svm.model avgpool18/ 256 labels/test_for_student.label soundnet.svm.csv
SoundNet with MLP
$ python3 test_mlp.py models/soundnet-avgpool18.mlp.model avgpool18/ 256 labels/test_for_student.label soundnet.mlp.csv
```

## For training
```
MFCC with SVM
$ python3 train_svm_multiclass.py bof/ 50 labels/train.csv labels/val.csv models/mfcc-50.svm.multiclass.model
MFCC with MLP
$ python3 train_mlp_mfcc.py bof/ 50 labels/train.csv labels/val.csv models/mfcc-50.mlp.model
SoundNet with SVM
$ python3 train_svm_multiclass.py avgpool18/ 256 labels/train.csv labels/val.csv models/soundnet-avgpool18.svm.model
SoundNet with MLP
$ python3 train_mlp.py avgpool18/ 256 labels/train.csv labels/val.csv models/soundnet-avgpool18.mlp.model
```


