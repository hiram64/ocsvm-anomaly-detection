# Unsupervised Anomaly Detection with One-Class Support Vector Machine

This repository includes codes for unsupervised anomaly detection by means of One-Class SVM(Support Vector Machine). In the codes, CIFAR10 is expected to be used.
Firstly, the image data are compressed by convolutional autoencoder(CAE) to vector features. Secondly, training a model only with the features of the data which you define as normal will be done. At the last, you can run anomaly detection with One-Class SVM and you can evaluate the models by AUCs of ROC and PR.

#### Dependencies
scikit-learn, Keras, Numpy, OpenCV

My test environment: Python3.6, scikit-learn==.21.2, Keras==2.2.4
, numpy==1.16.4, opencv-python==4.1.0.25

## How to use
### 1. Prepare data
Prepare data and labels to use. For instance, CIFAR10 is composed of 10 classes and each label should express unique class and be integer. These prepared data should be placed in the data directory.

You can download CIFAR10 data via :  
https://www.kaggle.com/janzenliu/cifar-10-batches-py

Put them in "data" directory and run the following code to compress them into NPZ file.
```
python make_cifar10_npz.py
```
After running this code, you can get cifar10.npz under "data" directory.


#### (Optional)
When you use your own dataset, please prepare npz file as the same format as CIFAR-10.
```
data = np.load('your_data.npz')
data.files
-> ['images', 'labels'] # "images" and "labels" keys'

data['labels']
-> array([6, 9, 9, ..., 5, 1, 7]) # labels is the vector composed of integers which correspond to each class identifier.

Note : Please be careful fo input image size of model.py.
You might need to change network architecture's parameter so that it can deal with your images.
```


### 2. Train CAE
Run the following command. Settable parameters like epoch, batchsize or output directory are described in the script.
```
python cae.py
```
The encoded features by CAE will be saved in the "data" directory as cifar10_cae.npz.

### 3. Run Anomaly Detection
First, normal class needs to be defined by "normal_label". It means the other classes EXCEPT the normal class will be automatically defined as abnormal.
By running the script below, OC-SVM is trained with the normal data. As evaluation metrics, AUCs of ROC(Receiver Operating Characteristic) and PR(Precision and Recall) are calculated.

By default, training models and test procedure are repeated over different nu parameters(see scikit-learn document. gamma and kernel are fixed in the script). For each nu and its trained model, the AUCs are averaged over 10 different test data set.

```
python anomaly_detection_ocsvm.py
```
Please look into the script for the settable parameters.

scikit-learn(sklearn.svm.OneClassSVM)  
http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
