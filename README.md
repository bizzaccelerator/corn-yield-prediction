# Corn yield prediction service in Kenia

*A Python-based predictive application for estimating corn yields using survey data, containerized with Docker.*

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

The proposed engineering solution is based on an `Optimized Gradient Boosted Tree model`, achieving an average deviation of 41.775 units from the test values and explaining 90.14% of the variability in corn yield production. This model outperformed other algorithms tested.

The model was selected after an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/notebook.ipynb).

The solution is implemented as a Python-based predictive service designed to estimate corn yields using survey data from farmers. It is deployed as a web application, enabling office teams to process survey data and predict expected corn yields for the current season, so they can take actions to reduce food insecurity in the county.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed were the Linear, Ridge and Lasso Regression; the second group studied were the Random Forest and it's the optimized version, and finally, the Gradient Boosted Trees and its Optimized version were taken into account too. An Optimized Gradient Boosted Tree model was chosen after evaluating various algorithms for its superior performance in balancing prediction accuracy and interpretability. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/japondo/corn-farming-data). However, a copy of the referred data is added to this repository for convenience. 

The application is built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance. 

## How to run the project.

Follow the steps below to reproduce the project locally or on a remote environment:

### _1. Prerequisites:_
Ensure the following are installed on your system:

- Git - Download Git
- Docker - Install Docker
- (Optional) Python - If you'd like to run the application without Docker.

### _2. Clone the Repository:_

- Open a terminal and navigate to the desired folder.
- Clone the repository:

> git clone https://github.com/your-username/your-repo-name.git cd your-repo-name

### _3. Build the Docker Image:_

- Ensure Docker is running.
- Build the Docker image:

> docker build -t corn-yield-prediction:latest .

### _4. Run the Application:_

- Start a container:

> docker run -d -p 9696:9696 --name corn-yield-app corn-yield-prediction:latest

### _5. Access the application:_

- Open your web browser and go to: http://localhost:9696
- Alternatively, test with curl or Postman.

### _6. Stopping and Removing the Container:_

- To stop the container:

> docker stop corn-yield-app

- To remove the container:

> docker rm corn-yield-app
