# Drowsiness Detector

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)
3. [Hypothesis and Validation](#hypothesis-and-validation)
4. [Model Rationale](#model-rationale)
5. [Business Requirements Rationale and Mapping](#business-requirements-rationale-and-mapping)
6. [Machine Learning Business Case](#machine-learning-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Unfixed Bugs](#unfixed-bugs)
9. [Deployment](#deployment)
10. [Technologies Used](#technologies-used)
11. [Credits](#credits)
12. [Acknowledgments](#acknowledgements)

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

Convolutional Layers:

The model has three convolutional layers which are instrumental for analyzing image data because they capture spatial hierarchies of features.
With multiple convolutional layers, the model learns increasingly complex features. The initial layers likely detect basic features like edges and textures. Deeper layers can recognize more intricate patterns that might be directly associated with the state of the eye (drowsy or awake).

Activation Function - ReLU:

The ReLU activation function introduces non-linearity in a computationally efficient manner.
Using ReLU helps prevent the vanishing gradient problem during backpropagation.

MaxPooling Layers:

After each convolutional layer, a MaxPooling layer is used to downsample the feature maps.
This reduces computational requirements and captures the essential features.

Flatten Layer:

This layer is used to transform the 3D output from preceding layers into a 1D vector, suitable for dense layers.

Dense Layers and Hyperparameter hp_units:

Based on the results of the hyperparameter search, the optimal number of units in the dense layer was found to be 320.
This specific choice strikes a balance between model complexity and the risk of overfitting. Having 320 units allows the model to capture a good amount of information without becoming overly complex.

Dropout Layer:

A rate of 0.5 means approximately half of the input units to this layer will be dropped out at each training step, promoting generalization and preventing overfitting.

Output Layer:

The model uses a sigmoid activation function, ideal for binary classification (awake or drowsy).

Learning Rate and Hyperparameter hp_learning_rate:

The optimal learning rate for the optimizer, based on the search results, is 0.001.
This learning rate provides a balance between convergence speed and the risk of overshooting the optimal values during training. The result from the hyperparameter search suggests that a learning rate of 0.001 allows for stable and effective training on the given dataset.

Optimizer - Adam:

Adam is an effective choice due to its adaptive learning rate properties. It adjusts the learning rate for each parameter, facilitating faster convergence without overshooting.

Loss Function - Binary Crossentropy:

Suitable for binary classification, it measures the difference between the actual and predicted probabilities.

The results from the hyperparameter tuning, specifically the selection of 320 units for the dense layer and a learning rate of 0.001, indicate the configurations that provided the best performance on the validation data for this specific task of differentiating between awake and drowsy eye images. This optimization ensures that the model is neither too simple (and underfits the data) nor too complex (and overfits), and it learns at an optimal pace given the data's characteristics.

## Business Requirements Rationale and Mapping
* List your business requirements and a rationale to map them to the Data Visualizations and ML tasks


## Machine Learning Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Technologies Used
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements
* Thank the people that provided support through this project.

