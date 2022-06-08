# Detection-of-intracranial-hemorrhage-on-CT-images-using-deep-learning

Intracranial hemorrhage (ICH) is bleeding between the brain tissue and skull or within the brain tissue itself, which is mainly liable for stroke. It is a life- threatening 
emergency. X-rays and computed tomography (CT) scans are widely applied for locating the hemorrhage position and size. This project 
presents an automated intracranial hemorrhage diagnosis using Convolutional Neural Network (CNN) with global average pooling. The obtained DICOM images 
are preprocessed and trained using CNN with 3 layers having 16, 64 and 128 layers each. Output layer of 5 nodes is extracted for 5 classes of Intracranial Hemorrhage 
with Sigmoid activation function. This helps to detect the healthy brain or five different types of intracranial hemorrhage. The obtained accuracy is 63.37% which 
can be further improved by increasing the number of training data set and increasing the size of the model. 

## How to use
In order to use this project you need to get the data set which can be obtained from kaggele. Then you need to preprocess it and train it yourself as the model obtained hasn't been uploaded. While running it mind the paths.

## Tools Used
* Python
* Pydicom
* NumPy
* Tensorflow
* Pandas
* SciPy
* Matplotlib
* OpenCV

## Images
!![image](https://user-images.githubusercontent.com/58971643/171201887-646842e8-6c6e-465a-88b8-7cbe872b232e.png)
!![image](https://user-images.githubusercontent.com/58971643/171201943-a3f2e29c-5745-4982-b516-f45cd2704b0a.png)
!![image](https://user-images.githubusercontent.com/58971643/171201974-c22622cd-e1a9-4ccb-b885-54ae21fd512c.png)

