# Drowsiness Detector

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Model Rationale](#model-rationale)
5. [User Stories](#user-stories)
6. [Business Requirements Rationale and Mapping](#business-requirements-rationale-and-mapping)
7. [Machine Learning Business Case](#machine-learning-business-case)
8. [Dashboard Design](#dashboard-design)
9. [Unfixed Bugs](#unfixed-bugs)
10. [Deployment](#deployment)
11. [Technologies Used](#technologies-used)
12. [Credits](#credits)
13. [Acknowledgments](#acknowledgements)

### Deployed Dashboard [here](https://drowsiness-detector-2d6e5a9a5e32.herokuapp.com/)


## Dataset Content
The dataset contains 9869 featured photos of computer generated eyes showcasing various stages of drowsiness against a neutral background, and also fully alert and awake eyes. Drowsiness while driving is a perilous condition that can lead to severe accidents and loss of lives. It can affect anyone behind the wheel but [studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1757738/) show that a large part of vehicular accidents due to tiredness or drowsiness occurs with occupational drivers such as the drivers of lorries, goods vehicles, and company cars. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/hazemfahmy/openned-closed-eyes).


## Business Requirements
In an era where the number of vehicular accidents due to drowsy driving is on the rise, EyeVigil Systems Inc. has identified a critical need for a state-of-the-art machine learning system. We have been hired by EyeVigil Systems to provide a 'proof of concept' showing that machine learning is capable of analysing the human eye, and based upon its 'openness' determine if it is showing signs of drowsiness. This project has three Business Requirements, a study to visually identify drowsy drivers, accurate prediction of driver drowsiness, downloadable drowsiness prediction report. Successfully fulfilling these will provide proof of concept and thus allow for further development, with the end goal of providing a real-time video machine learning system to detect drowsiness. Below the Business Requirements are outlined further.

1. A Study to Visually Identify Drowsy Drivers:

The system will perform image analysis on images to visually differentiate between alert and drowsy drivers. It will study the 'openness' of drivers' eyes to identify signs of drowsiness.

2. Accurate Prediction of Driver Drowsiness:

The drowsiness detection model will be developed as a binary classifier to accurately predict whether a given driver is drowsy or not based on their eye images. The model will utilize machine learning algorithms and computer vision techniques to achieve high accuracy in identifying drowsiness in drivers.

3. Downloadable Drowsiness Prediction Reports:

Upon analyzing a driver's eye images, the system will generate a comprehensive prediction report for each examination. The report will include details such as the date and time of the examination, the prediction result (Drowsy or Awake), and the associated probability. The report will be downloadable in a user-friendly format for record-keeping and further analysis.


## Hypothesis and Validation
* Hypothesis 1:

Drowsy individuals tend to have more closed eyes compared to alert individuals.

* Validation:

To validate this hypothesis, we analyze the average eye openness of drowsy and alert individuals from the dataset. Using Machine Learning we should be able to determine that the average openness of drowsy eyes is significantly lower than alert eyes. We can see this through the Average Image and Variability Images.

* Hypothesis 2:

There is a visual pattern that can be learned by a machine learning model to classify drowsy and alert eyes with an accuracy of at least 90%.

* Validation:

To validate this hypothesis, we train a machine learning model on the dataset and evaluated its performance. Should the model achieve an accuracy greater than 90% for the test set, then the client will consider this to be a validated hypothesis. 


## Model Rationale

To help choose the optimal hyperparameters I implemented the uses of a Keras Tuner. Below I will go through each of the hyperparameters in detail. 

* Convolutional Layers

The model has three convolutional layers which are instrumental for analyzing image data because they capture spatial hierarchies of features.
With multiple convolutional layers, the model learns increasingly complex features. The initial layers likely detect basic features like edges and textures. Deeper layers can recognize more intricate patterns that might be directly associated with the state of the eye (drowsy or awake).

* Activation Function - ReLU:

The ReLU activation function introduces non-linearity in a computationally efficient manner.
Using ReLU helps prevent the vanishing gradient problem during backpropagation.

* MaxPooling Layers:

After each convolutional layer, a MaxPooling layer is used to downsample the feature maps.
This reduces computational requirements and captures the essential features.

* Flatten Layer:

This layer is used to transform the 3D output from preceding layers into a 1D vector, suitable for dense layers.

* Dense Layers and Hyperparameter hp_units:

Based on the results of the hyperparameter search, the optimal number of units in the dense layer was found to be 320.
This specific choice strikes a balance between model complexity and the risk of overfitting. Having 320 units allows the model to capture a good amount of information without becoming overly complex.

* Dropout Layer:

A rate of 0.5 means approximately half of the input units to this layer will be dropped out at each training step, promoting generalization and preventing overfitting.

* Output Layer:

The model uses a sigmoid activation function, ideal for binary classification (awake or drowsy).

* Learning Rate and Hyperparameter hp_learning_rate:

The optimal learning rate for the optimizer, based on the search results, is 0.001.
This learning rate provides a balance between convergence speed and the risk of overshooting the optimal values during training. The result from the hyperparameter search suggests that a learning rate of 0.001 allows for stable and effective training on the given dataset.

* Optimizer - Adam:

Adam is an effective choice due to its adaptive learning rate properties. It adjusts the learning rate for each parameter, facilitating faster convergence without overshooting.

* Loss Function - Binary Crossentropy:

Suitable for binary classification, it measures the difference between the actual and predicted probabilities.

The results from the hyperparameter tuning, specifically the selection of 320 units for the dense layer and a learning rate of 0.001, indicate the configurations that provided the best performance on the validation data for this specific task of differentiating between awake and drowsy eye images. This optimization ensures that the model is neither too simple (and underfits the data) nor too complex (and overfits), and it learns at an optimal pace given the data's characteristics.

## User Stories
1. As a client, I want to view average images and image variances for 'awake' and 'drowsy' eyes, so I can determine the visual difference between the two. 
2. As a client, I want to see a collection of images from each category so I can see a standard collection of the categories.
3. As a client, I want to be able to access a machine learning tool, so that I can aquire a predicted state of an eye based on the image provided.
4. As a client, I want to be able to view the prediction probabilty, so that I can assess the potential accuracy.
5. As a client, I want the machine learning tool to have an accuacy of at least 90%, so that I can accertain if this model can be developed further.
6. As a client, I want to be able to upload multiple images, so that I can have a report generated for multiple images at one time. 
7. As a client, I want to be able to easily download a report of the model prediction, so I can record the predictions that have been made. 

## Business Requirements Rationale and Mapping
### Business Requirement 1: A Study to Visually Identify Drowsy Drivers
The system will perform image analysis on images to visually differentiate between alert and drowsy drivers. It will study the 'openness' of drivers' eyes to identify signs of drowsiness.
- As a client, I want to view average images and image variances for 'awake' and 'drowsy' eyes, so I can determine the visual difference between the two. 
- As a client, I want to see a collection of images from each category so I can see a standard collection of the categories. 

The User Stories above were addressed in the implementation of the following...
 1. The Data Visualation page in the Streamlit dashboard web tool developed. 
 2. Within the Data Visualization page the user can see both Average Variabily and Mean and Difference between the two categories. 
 3. Also within the Data Visualization page the user can view an image of a montage of each category, comprised of random images from each. 
### Business Requirement 2: Accurate Prediction of Driver Drowsiness
The drowsiness detection model will be developed as a binary classifier to accurately predict whether a given driver is drowsy or not based on their eye images. The model will utilize machine learning algorithms and data analytic techniques to achieve high accuracy, at least 90%, in identifying drowsiness in drivers.
- As a client, I want to be able to access a machine learning tool, so that I can aquire a predicted state of an eye based on the image provided.
- As a client, I want to be able to view the prediction probabilty, so that I can assess the potential accuracy.
- As a client, I want the machine learning tool to have an accuacy of at least 90%, so that I can accertain if this model can be developed further.
- As a client, I want to be able to upload multiple images, so that I can have a report generated for multiple images at one time.

The User Stories above were addressed in the implemntation of the following...
 1. The Drowsiness Detector page allows for users to upload images, single or multiple at a time.
 2. The paage, once an image has been uploaded, displays the prediction under the image. 
 3. A graph is provided that shows the percenage of probabilyt for an accurate prediction. 
 4. The Project Machine Learning Performance page has a table that shows the Loss and Accuracy of the model.
### Business Requirement 3: Downloadable Drowsiness Prediction Reports
Upon analyzing a driver's eye images, the system will generate a comprehensive prediction report for each examination. The report will include details such as the date and time of the examination, the prediction result (Drowsy or Awake), and the associated probability. The report will be downloadable in a user-friendly format for record-keeping and further analysis.
- As a client, I want to be able to easily download a report of the model prediction, so I can record the predictions that have been made. 

The User Stories above were addressed in the implemntation of the following...
 1. The Drowsiness Detector page allows for users to download the full report of the predictions for any and all of the images they upload to the detector rool

## Machine Learning Business Case

* The goal of this project is to leverage machine learning to ultimately create a drowsiness detector that can alert drivers when signs of drowsiness are detected, ensuring roads are safer. The drowsiness detector will also be a two/multi-class, single-label, classification model where the two primary classes would be "Awake" and "Drowsy." 

* Our ideal outcome is to provide a reliable proof of concept, that a machine learning tool can detect early signs of drowsiness and differentiate between it and alert. 

* Success Metrics: The model should have an cccuracy of 90% or above on the test set.

* The output should be a clear indication of whether the person is "Awake" or "Drowsy" along with the associated probability of the prediction. This real-time prediction will provide immediate feedback to the user.

* Currently, many vehicles come with basic drowsiness alert systems that often use steering pattern recognition. However, this method is not foolproof and often results in false alarms or misses real drowsiness events. Other methods include manual self-assessment by drivers, but this is highly unreliable as a person's judgment may be impaired when they are drowsy. If this project provides proof of concept further development will result in a realtime video analysis of the drivers.

* The dataset used for this proof of concept was provided from Kaggle, linked in the Dataset Content section of this ReadMe file. 

## Dashboard Design
The dashboard for this project was developed using Streamlit. It consists of five pages, Project Summary, Drowsiness Visualization, Drowsiness Detector, Project Hypothesis, and Project Machine Learning Performance. 
### Home Page-Project Summary
This page contains information about the project such as the general purpose behind the project, information about the dataset used, and a list of the business requirements. 
<details><summary>Show Project Summary</summary>
<img src="readme_imgs/page1.jpg">
</details>

### Drowsiness Visualization
The Drowsiness Visualization page first displays the variability between 'awake' and 'drowsy' eyes while also showing the mean or average image of each, and displaying the difference between those averages. 
<details><summary>Show Average and Variabilty</summary>
<img src="readme_imgs/page2.2.jpg">
</details>
<details><summary>Show Average 'Awake' and 'Drowsy', and Difference</summary>
<img src="readme_imgs/page2.3.jpg">
</details>
<br>
This page also displays a montage of random images fro both Awake and Drowsy labels, allowing the user to choose in a dropdown which of the two they want to view.
<details><summary>Show Montage</summary>
<img src="readme_imgs/page2.4.jpg">
</details>

### Drowsiness Detector
This page allows users to upload images from the dataset to test if the eye is either 'Awake' or 'Drowsy'. A link is provided to the original dataset alowing the user to download the images for the detector to use. Once uploaded the page will make an analysis and prediction of its state, either 'Awake' or 'Drowsy'.
<details><summary>Show Detector</summary>
<img src="readme_imgs/page3.1.jpg">
</details>
Below you can see an image uploaded for evaluation. It is a single image, but multiple can be uploaded as well. 
<details><summary>Show Montage</summary>
<img src="readme_imgs/page3.2.jpg">
</details>
After upload, a pridiction will be made. It will also display the probabilty of the accuracy of the predicion and for each image provide a report. The report is also downloadable by clicking the link 'Download Report'. 
<details><summary>Show Prediction</summary>
<img src="readme_imgs/page3.3.jpg">
</details>

### Project Hypothesis
This page displays the projects hypotheses and the conclusion in regards to the finds and model training. 
<details><summary>Show Hypotheses</summary>
<img src="readme_imgs/page4.jpg">
</details>

### Project Machine Learning Performance
This page displays the distribution of the data, as well explaining the different types of accuracy and loss and plotting accordingly, also providing a confusion matrix and other performance metrics. 

The graph shows the total number of images in the data set and how they have been divided between test, train, and validate, and the ratio of the division. 
<details><summary>Show Image Distribution</summary>
<img src="readme_imgs/page5.1.jpg">
</details>
There are two graphs that are showing the model performance in terms of the Accuracy and Loss.
<details><summary>Show Accuracy</summary>
<img src="readme_imgs/page5.2.jpg">
</details>
<details><summary>Show Loss</summary>
<img src="readme_imgs/page5.3.jpg">
</details>
The final graph on the page is the confusion matrix, as well as an explanation of the graph. There is also a small chart that shows the general performance of the model in terms of Loss and Accuracy. 
<details><summary>Show Prediction</summary>
<img src="readme_imgs/page5.4.jpg">
</details>

## Unfixed Bugs
There are currently no unfixed or known bugs. 

## Deployment
### Heroku

* The App live link is: https://drowsiness-detector-2d6e5a9a5e32.herokuapp.com/
* The project was deployed to Heroku using the following steps.

1. Create a requirement.txt file in GitHub, for Heroku to read, listing the dependencies the program needs in order to run.
2. Set the runtime.txt Python version to a version that the current Heroku stack supports.
3. Push the recent changes to GitHub and go to your Heroku account page to create and deploy.
4. Chose "CREATE NEW APP", give it a unique name, and select a geographical region.
5. From the Deploy tab, chose GitHub as deployment method, connect to GitHub and search for and select the project's repository.
6. Select the branch you want to deploy, then click Deploy Branch.
7. Click to "Enable Automatic Deploys " or chose to "Deploy Branch" from the Manual Deploy section.
8. Wait for the logs to run while the dependencies are installed and the app is being built.
9. The mock terminal is then ready and accessible from a link similar to https://your-projects-name.herokuapp.com/
10. If the slug size is too large then add large files not required for the app to the .slugignore file, similar to the .gitignore file.


### Forking the GitHub Project
By forking this GitHub Repository you make a copy of the original repository on our GitHub account to view and/or make changes without affecting the original repository. The steps to fork the repository are as follows:
1. Locate the GitHub Repository of this project and log into your GitHub account.
2. Click on the "Fork" button, on the top right of the page, just above the "Settings".
3. Decide where to fork the repository (your account for instance)
4. You now have a copy of the original repository in your GitHub account.

### Making a Local Clone
Cloning a repository pulls down a full copy of all the repository data that GitHub.com has at that point in time, including all versions of every file and folder for the project. The steps to clone a repository are as follows:
1. Locate the GitHub Repository of this project and log into your GitHub account.
2. Click on the "Code" button.
3. Chose one of the available options: Clone with HTTPS, Open with Git Hub desktop, Download ZIP.
4. To clone the repository using HTTPS, under "Clone with HTTPS", copy the link.
5. Open Git Bash, if you don't have it downloaded to your local machine do that now.
6. In the Git terminal type: git clone https://git.heroku.com/ml_pp5_drowsiness_detector.git
7. Press Enter, and wait for the repository to be created.
8. Open your prefered coding software of choice. 
9. Open the folder at the location, and the software should place all the folders and files from the repo no wihtin your coding environment. 

## Technologies Used
### Platforms
* Heorku
* Jupyter Notebooks
* Kaggle
* GitHub
* VSCode 
### Languages
* Python
* Markdown
### Data Analysis and Machine Learning Libraries
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* Scikit-learn
* Tensorflow
* Keras

## Credits 

### Content 
* Dataset from user [hazemfahmy](https://www.kaggle.com/hazemfahmy) on Kaggle

### Code
* Template used belongs to [CodeInstitute](https://github.com/Code-Institute-Solutions/milestone-project-bring-your-own-data)
* Walkthrough Project 1 was used as the skeletal structure of this project
* Keras Tuning learned from [TensorFlow](https://www.tensorflow.org/tutorials/keras/keras_tuner) resources



## Acknowledgements
This is my final project, and as such I have many people to thank...
 * My wonderful partner, Leo. Through this entire process you have been fighting in my corner and cheering me on. You have always believed I can do anything and you have been my rock. I love you, and thank you.
 * My supportive housemate, Trevor. You have stepped up many times to help where you shouldn't need to but did anyway. You have supported me and Leo through a lot this last year. Thank you for being dependable and helping. 
 * My mentor, Mo. You are a rockstar. You have taken time out of your busy day to help encourage me and gas me up to tackle things I didn't think I could. You have been everything a person coudl want in a mentor and I appreciate every ounce of energy and advise you gave me. Thank you so much.
 * The my friends, Oskar, Patrik, and Mark, you all have been so supportive and encouraging and understanding when I have to be locked away for weeks typing away instead of hanging out. But, you all believed in me and made me feel like I could do it. Thank you.  

