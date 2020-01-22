# Depth Image Homography Estimation with Noise in Pytorch

Pytorch implementation of the paper "Deep Image Homography Estimation” written by DeTone, Malisiewicz and Rabinovich. In this project it is possible to check the performance of the model against various types of noise effects (Blur 5x5, Blur10x10, Gaussian, Compression and Salt&Pepper).

### Links
* [Deep Homography Estimation paper](https://arxiv.org/abs/1606.03798)
* [MS-COCO Dataset](http://cocodataset.org/#download)

### How to run
* Download the MS-COCO dataset from the [official website](http://cocodataset.org/#download). Extract the data in the folder *data*. The expected folder structure is as follows:
```
project
│   README.md
│   train.py    
│   model.py    
│   dataset.py    
│
└───data
│   └───val201X
│   └───test201X
│   └───train201X
```

* Run **python3 train.py** to train and test the model.
    * If you want to train and and test the model against all type of noise you have to set the argument *--noise "All"*. Otherwise tou can select one between "Vanilla", "Compression", "Gaussian", "Blur5", "Blur10" and "S&P".