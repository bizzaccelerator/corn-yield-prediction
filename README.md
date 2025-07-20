# Corn yield prediction service in Kenia

*A Python-based predictive application for estimating corn yields using survey data, containerized with Docker.*

This fictional project was originally developed as a midterm evaluation for the Machine Learning Zoomcamp and improved as a final projecto for MLops Zoomcamp, both offered by Data Talks Club. Method and objectives were defined for educational purposes only, so I can show the knowledge appropiated during the mentioned training. 

The current project simulates a real scenario of information gathering to support effective political decision-making in a mayor's office in Kenya, aiming to ensure food security in the region. 

![CornField_Lead](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/CornField_Lead.jpg)
Photo: ©somkak – stock.adobe.com

## Problem statement
This could be understood in two leves: a business problem and a technical problem. 

### _Business problem:_
Certain region in Kenya has experienced rapid population growth over the past decade in an underdeveloped economic environment. The social group living in this region considers _corn_ as the preferred base for most typical dishes; however, the low level of precipitation threatens sufficient production in the coming years. The Mayor's Office seeks to make the best decisions to ensure food security in the county. To acheive that goal, the prediction of corn production at a household level is a must. That’s why the managing team at the political office needs to know the expected levels of corn production at a household level, the key variables affecting it, so they can further improve the resources allocation process.

### _Technical problem:_
As a senior Machine Learning Engineer, I am responsible for developing a model that predicts the amount of corn likely to be produced in a county in Kenya. This model is designed to support the mayor’s office in planning and distributing resources more effectively, helping improve outcomes for local agriculture.

To build the model, we use detailed data collected from various farms, including information such as the gender of the farm leader, household size, and fertilizer usage. These factors help us understand what influences corn yield.

Throughout the development process, we document the different approaches tested, the data used, and how the model performed. This helps ensure transparency and makes it easier to improve the model over time. We also store and manage different versions of the data and models, so we can always go back and compare results.

The final model is made available through a secure cloud platform, allowing decision-makers to access predictions and insights when needed. This setup supports both easy access and controlled permissions, ensuring the information is both useful and protected.

We’ve also put tools in place to monitor how the model performs in the real world. If the conditions that affect corn production change—such as weather patterns or farming practices—the system will alert us so the model can be reviewed and updated. This ongoing process ensures that the predictions remain accurate and that the tool continues to support strong, evidence-based decisions.

## Solution proposed

The proposed engineering solution is based on an `Optimized Gradient Boosted Tree model`, achieving an average deviation of 41.775 units from the test values and explaining 90.14% of the variability in corn yield production. This model outperformed other algorithms tested.

The model was selected after an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/notebook.ipynb).

The solution is implemented as a Python-based predictive service designed to estimate corn yields using survey data from farmers. It is deployed as a web application, enabling office teams to process survey data and predict expected corn yields for the current season, so they can take actions to reduce food insecurity in the county.

![Solution](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/Solution.JPG)
Photo: Diagram of the solution engineered.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed were the Linear, Ridge and Lasso Regression; the second group studied were the Random Forest and it's the optimized version, and finally, the Gradient Boosted Trees and its Optimized version were taken into account too. An Optimized Gradient Boosted Tree model was chosen after evaluating various algorithms for its superior performance in balancing prediction accuracy and interpretability. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/japondo/corn-farming-data). However, a copy of the referred data is added to this repository for convenience. 

The application was coded in python using a distribution of Anaconda. Conda was used to manage isolated virtual environments and install all the packages needed without conflicts. This solution was built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance. 

## How to run the project.

Follow the steps in the [wiki](https://github.com/bizzaccelerator/corn-yield-prediction/wiki/Welcome-to-the-Corn-yield-prediction-service-in-Kenia-wiki!) to reproduce the project.
