# Unsupervised Anomaly Detection with One-Class Support Vector Machine

This repository includes codes for unsupervised anomaly detection by means of One-Class SVM. In the codes, CIFAR10 is expected to be used.  
Firstly, the image data are compressed by convolutional autoencoder(CAE) to vector features. Secondly, training model only with the features of the data which you define as normal will be done. At the last, you can run anomaly detection with model and you can evaluate the models by AUCs of ROC and PR.  

#### Dependencies
scikit-learn, keras, numpy

## How to use
### 1. Prepare data
Prepare data and label to use. For instance, CIFAR10 is composed of 10 classes and each label should express unique class and be integer. These prepared data should be placed in the data directory.

### 2. Train CAE
Run the following command. Settable parameters like epoch, batchsize or output directory are described in the script.
```
python cae.py
```
The encoded features by CAE will be saved in the data directory.

### 3. Run Anomaly Detection
First, normal class need to be defined in "normal_label". It means the other classes except the normal class will be automatically defined as abnormal.
By running the script below, OC-SVM is trained with the normal data. As evaluation metrics, AUCs of ROC(Receiver Operating Characteristic) and PR(Precision and Recall) are calculated.

By default, training model and test procedure is repeated over different nu parameters(see scikit-learn document, gamma and kernel are fixed in the script). For each nu and its trained model, the AUCs are averaged over 10 different test data set.

```
python anomaly_detection_ocsvm.py
```
Please look into the script for settable parameters.

scikit-learn(sklearn.svm.OneClassSVM)  
http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
