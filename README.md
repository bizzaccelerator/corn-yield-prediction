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
The county in Kenya is facing a potential food security risk due to rapid population growth, underdeveloped economic conditions, and declining precipitation levels. Since corn is the staple food in this region, the Mayor’s Office requires a reliable and scalable system to predict household-level corn production and identify the key variables influencing yield. These insights are essential for making data-driven decisions about resource allocation and agricultural planning.

From a technical perspective, this requires the development of a machine learning framework that can:

- Ingest and process heterogeneous farm-level data, including demographic, household, and agricultural practice variables.

- Handle limited or incomplete datasets by incorporating methods for data augmentation or synthetic data generation.

- Evaluate multiple modeling approaches to ensure accuracy, fairness, and robustness in yield predictions.

- Support experiment tracking and version control to ensure transparency and reproducibility across model iterations.

- Enable deployment to a secure, cloud-based environment, accessible to authorized stakeholders while maintaining data privacy.

- Provide mechanisms for continuous monitoring, detecting model drift when environmental or farming conditions change.

- Be scalable and cost-efficient, so it can adapt to different levels of demand and long-term operational needs.

The core challenge is therefore to design and implement a machine learning system that balances predictive accuracy, operational reliability, and long-term adaptability, so that the Mayor’s Office can depend on it as a decision-support tool for addressing food security in the region.

## Solution proposed

To provide the Mayor’s Office with reliable insights, I designed an end-to-end MLOps pipeline that ensures reproducibility, scalability, and monitoring across the full lifecycle of machine learning models.

For the data foundation, I worked with historical datasets sourced from [Kaggle](https://www.kaggle.com/datasets/japondo/corn-farming-data/data). Since real-world data can sometimes be limited or incomplete, I also generated synthetic datasets specifically for testing purposes. This combination gave us the necessary volume and diversity of records to properly train, validate, and stress-test the models before pushing them into production.

For model development and experimentation, I tested several algorithms including Linear Regression, Lasso, Ridge, and Gradient Boosted Trees (GBT). To systematically fine-tune performance, I applied the Hyperopt library for hyperparameter optimization. All experiments, metrics, and parameters were tracked with MLflow, which made it straightforward to compare results and select the most effective approach.

To ensure smooth execution, I implemented pipeline orchestration and automation using Kestra, deployed on a dedicated Google VM. Kestra handled the automation of data preparation, model training, evaluation, deployment, and monitoring steps. By reducing manual intervention, it allowed workflows to be repeatable, reliable, and fully automated.

For model monitoring and validation, I integrated Evidently AI. This tool continuously monitored both training (baseline) and testing datasets. It generated reports on model performance, data drift, and data quality, helping anticipate issues before they could affect predictions. Additionally, the Evidently UI was deployed in Cloud Run, giving decision-makers direct access to visual reports and enabling them to detect potential risks early.

Once the best model was finalized, it was packaged as a Docker image and stored in Google Container Registry (GCR). From there, it was deployed into Google Cloud Run, which provided a REST API endpoint for external use by the Mayor’s Office or other systems. This design ensured the solution was scalable, serverless, and cost-efficient, adapting seamlessly to fluctuating demand.

The underlying infrastructure and CI/CD pipelines were fully managed through Terraform, which guaranteed reproducibility, consistency, and version control. To streamline deployments, I set up GitHub Actions, which automated testing, building, and rollout. Furthermore, integration testing with Docker Compose validated that services such as MLflow, Evidently, the database, and the orchestrator worked together correctly before any updates were released to Google Cloud.

Finally, for storage and artifact management, I established a dedicated structure in Google Cloud Storage. MLflow artifacts—including models, metrics, and parameters—were stored separately from Evidently reports. This separation supported strong data governance practices, ensured traceability of all model versions and evaluations, and simplified long-term maintenance.

![mlops_infra](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/mlops_infra.JPG)
Photo: Diagram of the technical infrastructure engineered.

The proposed engineering solution is based on an `Optimized Gradient Boosted Tree model`, achieving an average deviation of 41.775 units from the test values and explaining 90.14% of the variability in corn yield production. This model outperformed other algorithms tested.

The model was selected after an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/notebook.ipynb).

The solution is implemented as a Python-based predictive service designed to estimate corn yields using survey data from farmers. It is deployed as a web application, enabling office teams to process survey data and predict expected corn yields for the current season, so they can take actions to reduce food insecurity in the county.

![Solution](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/Solution.JPG)
Photo: Diagram of the prediction service disposed.

### _Rationale behind the solution:_ 

During the process different algorithms were tested. The first group analyzed were the Linear, Ridge and Lasso Regression; the second group studied were the Random Forest and it's the optimized version, and finally, the Gradient Boosted Trees and its Optimized version were taken into account too. An Optimized Gradient Boosted Tree model was chosen after evaluating various algorithms for its superior performance in balancing prediction accuracy and interpretability. 

The data used in this project was obtained for free from kagle [here](https://www.kaggle.com/datasets/japondo/corn-farming-data). However, a copy of the referred data is added to this repository for convenience. 

The application was coded in python using a distribution of Anaconda. Conda was used to manage isolated virtual environments and install all the packages needed without conflicts. This solution was built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance. 

## How to run the project.

Follow the steps in the [wiki](https://github.com/bizzaccelerator/corn-yield-prediction/wiki/Welcome-to-the-Corn-yield-prediction-service-in-Kenia-wiki!) to reproduce the project.
