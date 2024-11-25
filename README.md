# Corn yield prediction in Kenia

This fictional project was developed as a midterm evaluation for the Machine Learning Zoomcamp offered by Data Talks Club. Method and objectives were defined for educational purposes only, so I can show the knowledge appropiated during the firsts part of the training. 

The current project simulates a real scenario of information gathering to support effective political decision-making in a mayor's office in Kenya, aiming to ensure food security in the region. 

![CornField_Lead](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/CornField_Lead.jpg)
Photo: ©somkak – stock.adobe.com

## Problem statement
This could be understood in two leves: a business problem and a technical problem. 

### _Business problem:_
Certain region in Kenya has experienced rapid population growth over the past decade in an underdeveloped economic environment. The social group living in this region considers _corn_ as the preferred base for most typical dishes; however, the low level of precipitation threatens sufficient production in the coming years. The Mayor's Office seeks to make the best decisions to ensure food security in the county. To acheive that goal, the prediction of corn production at a household level is a must. That’s why the managing team at the political office needs to know the expected levels of corn production at a household level, the key variables affecting it, so they can further improve the resources allocation process.

### _Technical problem:_
As a Machine Learning engineer, I am tasked with building a model that not only predicts the amount of corn produced in that county in Kenya but also helps decision makers at the mayor's office organize resources for optimized corn production. To achieve this goal, the model construction explores data collected from various plantations, identifying several useful variables such as the gender of the plantation leader, the size of the household, and the amount of fertilizer used in corn production, among others. This model is implemented in a cloud solution that serves the model for future use and insights extraction, enhancing its reliability, readability, and security.

## Solution proposed
The engineering solution proposed was built over an `Optimized Gradient Boosted Tree` with an average deviation from the test values of 41.775 units of corn predicted, and provides a very high explanation of variability in yield production (90.1378%) amongs the different algorithms tested.

This model was selected after performing a profound Exploratory Data Analysis (EDA) where missing values, univariate distribuitions and feature importance analysis were made. Extensive EDA can be found [here](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/notebook.ipynb).

The solution is a Python-based predictive service designed to estimate corn yields using survey data provided by farmers. This service is served as a web application that could be used by the office team to process information from the survey applied to farmers, returning the predicted amount of corn yield expected to obtain during this season.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed were the Linear, Ridge and Lasso Regression; the second group studied were the Random Forest and it's the optimized version, and finally, the Gradient Boosted Trees and its Optimized version were taken into account too. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/japondo/corn-farming-data). However, a copy of the referred data is added to this repository for convenience. 

The model is served as a web application using FLask through a docker container. The library 

## How to run the project.
