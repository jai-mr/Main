### Repository github url : https://github.com/jai-mr/Assignments
### Assignment Repository : https://github.com/jai-mr/Assignments/tree/main/Assignment-8
### Main Repository : https://github.com/jai-mr/Main
### Submitted by : Jaideep R - No Partners
### Registered email id : jaideepmr@gmail.com

## Problem Statement
Train a ResNet18 model on CIFAR10 dataset for 20 epochs
### Main Repository
* https://github.com/jai-mr/Main
### Assignment Repository 
* To be used in all subsequent assignments

```text
├── main.py
├── models
│   └── resnet.py
├── README.md
├── requirements.txt
└── utils
    ├── grad_cam.py
    ├── helper.py
    ├── plot_utils.py
    ├── test.py
    ├── train.py
    └── transforms.py
```

#ResNet Model summary


```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         GroupNorm-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
         GroupNorm-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
         GroupNorm-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
         GroupNorm-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
        GroupNorm-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
        GroupNorm-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
        GroupNorm-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
        GroupNorm-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
        GroupNorm-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
        GroupNorm-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
        GroupNorm-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
        GroupNorm-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
        GroupNorm-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
        GroupNorm-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
        GroupNorm-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
        GroupNorm-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
        GroupNorm-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
        GroupNorm-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
        GroupNorm-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
        GroupNorm-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 13.50
Params size (MB): 42.63
Estimated Total Size (MB): 56.14
----------------------------------------------------------------

```

### Training and Testing logs

from [Assignment-8.ipynb](https://github.com/jai-mr/Assignments/blob/main/Assignment-8/Assignment-8.ipynb)

* Training was done for 20 epochs.
* Needed more time to check on all aspects to improve the test accuracy with the various combinations. 

```text

Epoch 1:
Train Loss=2.3361639976501465 Batch_id=390 LR= 0.10436 Train Accuracy= 10.07: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 2:
Train Loss=2.3169546127319336 Batch_id=390 LR= 0.28019 Train Accuracy= 9.89: 100%|██████████| 391/391 [04:10<00:00,  1.56it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 3:
Train Loss=2.3022003173828125 Batch_id=390 LR= 0.52032 Train Accuracy= 10.07: 100%|██████████| 391/391 [04:10<00:00,  1.56it/s]

: Average Test loss: 0.0184, Test Accuracy: 1000/10000 (10.00%)

Epoch 4:
Train Loss=2.342921733856201 Batch_id=390 LR= 0.76037 Train Accuracy= 9.98: 100%|██████████| 391/391 [04:09<00:00,  1.57it/s]

: Average Test loss: 0.0187, Test Accuracy: 1000/10000 (10.00%)

Epoch 5:
Train Loss=2.36979341506958 Batch_id=390 LR= 0.93596 Train Accuracy= 9.98: 100%|██████████| 391/391 [04:09<00:00,  1.57it/s]

: Average Test loss: 0.0191, Test Accuracy: 1000/10000 (10.00%)

Epoch 6:
Train Loss=2.3799023628234863 Batch_id=390 LR= 1.00000 Train Accuracy= 10.16: 100%|██████████| 391/391 [04:10<00:00,  1.56it/s]

: Average Test loss: 0.0184, Test Accuracy: 1000/10000 (10.00%)

Epoch 7:
Train Loss=2.377091407775879 Batch_id=390 LR= 0.98740 Train Accuracy= 9.93: 100%|██████████| 391/391 [04:12<00:00,  1.55it/s]

: Average Test loss: 0.0188, Test Accuracy: 1000/10000 (10.00%)

Epoch 8:
Train Loss=2.3018593788146973 Batch_id=390 LR= 0.95036 Train Accuracy= 10.07: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0187, Test Accuracy: 1000/10000 (10.00%)

Epoch 9:
Train Loss=2.32676100730896 Batch_id=390 LR= 0.89074 Train Accuracy= 10.04: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0187, Test Accuracy: 1000/10000 (10.00%)

Epoch 10:
Train Loss=2.3184707164764404 Batch_id=390 LR= 0.81152 Train Accuracy= 10.11: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0188, Test Accuracy: 1000/10000 (10.00%)

Epoch 11:
Train Loss=2.3865301609039307 Batch_id=390 LR= 0.71668 Train Accuracy= 10.05: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 12:
Train Loss=2.3028602600097656 Batch_id=390 LR= 0.61098 Train Accuracy= 9.84: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 13:
Train Loss=2.295814037322998 Batch_id=390 LR= 0.49972 Train Accuracy= 9.90: 100%|██████████| 391/391 [04:11<00:00,  1.56it/s]

: Average Test loss: 0.0184, Test Accuracy: 1000/10000 (10.00%)

Epoch 14:
Train Loss=2.3309664726257324 Batch_id=390 LR= 0.38846 Train Accuracy= 10.00: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0182, Test Accuracy: 1000/10000 (10.00%)

Epoch 15:
Train Loss=2.29038667678833 Batch_id=390 LR= 0.28280 Train Accuracy= 9.66: 100%|██████████| 391/391 [04:11<00:00,  1.56it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 16:
Train Loss=2.3078880310058594 Batch_id=390 LR= 0.18803 Train Accuracy= 9.83: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 17:
Train Loss=2.3160862922668457 Batch_id=390 LR= 0.10891 Train Accuracy= 9.88: 100%|██████████| 391/391 [04:11<00:00,  1.56it/s]

: Average Test loss: 0.0183, Test Accuracy: 1000/10000 (10.00%)

Epoch 18:
Train Loss=2.3166158199310303 Batch_id=390 LR= 0.04939 Train Accuracy= 9.93: 100%|██████████| 391/391 [04:11<00:00,  1.56it/s]

: Average Test loss: 0.0182, Test Accuracy: 1000/10000 (10.00%)

Epoch 19:
Train Loss=2.302061080932617 Batch_id=390 LR= 0.01248 Train Accuracy= 9.97: 100%|██████████| 391/391 [04:12<00:00,  1.55it/s]

: Average Test loss: 0.0182, Test Accuracy: 1000/10000 (10.00%)

Epoch 20:
Train Loss=2.303800106048584 Batch_id=390 LR= 0.00000 Train Accuracy= 9.90: 100%|██████████| 391/391 [04:11<00:00,  1.55it/s]

: Average Test loss: 0.0182, Test Accuracy: 1000/10000 (10.00%)


```

### Misclassified images

from [Assignment-8.ipynb](https://github.com/jai-mr/Assignments/blob/main/Assignment-8/Assignment-8.ipynb)


![picture 1](images/misclassified1.png)  

### Plots for Train / Test Loss & Accuracy

from [Assignment-8.ipynb](https://github.com/jai-mr/Assignments/blob/main/Assignment-8/Assignment-8.ipynb)


![picture 2](images/metrics.png)  

### 1.4.6. GradCam output for misclassified images

from [Assignment-8.ipynb](https://github.com/jai-mr/Assignments/blob/main/Assignment-8/Assignment-8.ipynb)


![picture 3](images/gradcam1.png)  

![picture 4](images/gradcam2.png)  
