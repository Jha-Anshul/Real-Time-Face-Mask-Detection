# Real-Time-Face-Mask-Detection
Real Time Face Mask Detection Using Deep Learning and Computer Vision

**Introduction:**

To control the spread of COVID-19 pandemic World Heath Health Organization, Centre of Disease Control and Prevention and several governments across the World have made wearing a mask compulsory for public safety. Wearing a mask can decrease the risk of infection from the harmful virus and limit the spread of disease. However, it is difficult to keep check whether an individual is wearing a mask. Therefore, face mask detection is essential tasks to detect the individual wearing mask.


**Dataset:**

The dataset consists of 7553 images. The dataset is categorized into two folders with mask and without mask. The folder with mask consists of 3725 images of people faces with mask and 3828 images of people face without mask.

Dataset used in this project is available on Kaggle.

Link: https://www.kaggle.com/omkargurav/face-mask-dataset

**Required Libraries:**

1. Tensorflow
2. Keras
3. Numpy
4. Pandas
5. Matplotlib
6. CV2


**Project:**

The project is divided into two steps:
1. Train the face mask detector model using a convolutional neural network.
2. Apply the real-time face mask detection using openCV.

**How to run the project?**

1. Create a directory in your local machine and change ditectory 

    ` mkdir ~/Desktop/face_mask_detection_project                                             `

    ` cd ~/Desktop/face_mask_detection_project                                                `

2. Clone the repository and cd into the folder

    ` git clone https://https://github.com/Jha-Anshul/Real-Time-Face-Mask-Detection.git       `

    ` cd Real-Time-Face-Mask-Detection                                                        `

3. To start a video streaming and real time face mask detction, run the following command:

    `python3 video_streaming.py `                                                            

To quit the video streaming hit "q"

4. If you make changes in face_mask_detector.py then run the following command before step 3:

    ` python3 face_mask_detector.py`


**Results:**

_Data visulaization_

![image](https://user-images.githubusercontent.com/85516257/123308659-8a700e00-d4e9-11eb-9366-da99c7be0eb6.png)

_Predicted images by CNN model:_

![image](https://user-images.githubusercontent.com/85516257/123308718-9fe53800-d4e9-11eb-82a5-5ebe7ed82c1f.png)


_Plot of Accuracy and Loss_

![image](https://user-images.githubusercontent.com/85516257/123308846-cc994f80-d4e9-11eb-8c25-7933c6ca3929.png)

The training accuracy is 99.78% and validation accuracy is 96.42%

