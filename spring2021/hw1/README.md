# HW1

## For testing
Generate csv for testing mfcc and SoundNet with SVM and MLP classifier
```
$ python3 test_svm_multiclass.py models/mfcc-50.svm.multiclass.model bof/ 50 labels/test_for_student.label mfcc-50.svm.multiclass.csv

$ python3 test_mlp.py models/mfcc-50.mlp.model bof 50 labels/test_for_student.label mfcc-50.mlp.csv

$ python3 test_svm_multiclass.py models/soundnet-avgpool18.svm.model avgpool18/ 256 labels/test_for_student.label soundnet.svm.csv

$ python3 test_mlp.py models/soundnet-avgpool18.mlp.model avgpool18/ 256 labels/test_for_student.label soundnet.mlp.csv
```

## For training
```
MFCC with SVM
$ python3 train_svm_multiclass.py bof/ 50 labels/train.csv labels/val.csv models/mfcc-50.svm.multiclass.model

$ python3 train_mlp_mfcc.py bof/ 50 labels/train.csv labels/val.csv models/mfcc-50.mlp.model

$ python3 train_svm_multiclass.py avgpool18/ 256 labels/train.csv labels/val.csv models/soundnet-avgpool18.svm.model

$ python3 train_mlp.py avgpool18/ 256 labels/train.csv labels/val.csv models/soundnet-avgpool18.mlp.model
```


