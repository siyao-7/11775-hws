# HW3

## Feature extraction
SoundNet: Same as HW1 https://github.com/siyao-7/11775-hws/tree/master/spring2021/hw1. Put the final result under folder avgpool18/
I3D: follow instructions in https://github.com/Finspire13/pytorch-i3d-feature-extraction, put the result under folder i3d/

## For testing
Generate csv for testing Early Fusion and Late Fusion with SVM classifier
```
Early Fusion
$ python test_early_fusion.py models/early_fusion.model labels/test_for_student.label best.csv
Late Fusion
$ python test_late_fusion.py models/late_fusion.model labels/test_for_student.label late_fusion.csv
```

## For training
```
Early Fusion
$ python train_early_fusion.py models/early_fusion.model
Late Fusion
$ python train_late_fusion.py models/late_fusion.model
```


