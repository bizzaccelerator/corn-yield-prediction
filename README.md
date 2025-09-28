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

This project delivers an end-to-end MLOps pipeline designed to predict household-level corn yields and support food security decisions for the Mayor’s Office. The system integrates modern data engineering, machine learning, and deployment tools into a cohesive architecture.

The solution leverages the following core technologies and their relationships:

| **Component**                     | **Technology**       | **Purpose**                                                                 |
|-----------------------------------|----------------------|-----------------------------------------------------------------------------|
| Data & Experiment Tracking        | MLflow               | Manages experiment tracking, model registry, and artifact storage.          |
| Workflow Orchestration            | Kestra               | Coordinates data preparation, training, evaluation, deployment, and monitoring tasks. |
| Model Monitoring                  | Evidently AI         | Generates reports on data drift, performance, and data quality, providing insights during training and production. |
| Model Serving                     | Docker + Cloud Run   | Packages the best-performing model as a Docker image and deploys it in Google Cloud Run, exposing a REST API endpoint. |
| Infrastructure Management         | Terraform            | Provisions cloud resources (Cloud Run, GCS buckets, networking, VM for Kestra) to ensure reproducibility and scalability. |
| Continuous Integration & Delivery | GitHub Actions       | Automates testing, building, and deployment pipelines.                       |
| Storage                           | Google Cloud Storage | Separates MLflow artifacts (models, metrics, parameters) from Evidently reports to ensure governance and traceability. |

Together, these technologies form a modular, scalable, and cost-efficient pipeline that enables reliable corn yield predictions and continuous improvement of the model lifecycle.

![mlops_infra](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/mlops_infra.jpg)
Photo: Diagram of the technical infrastructure engineered.

### _Rationale behind the solution:_

The end-to-end pipeline was designed to ensure accuracy, transparency, and operational robustness from data ingestion to model serving.

For the data foundation, historical records were obtained from [kaggle](https://www.kaggle.com/datasets/japondo/corn-farming-data), complemented with synthetic datasets to overcome real-world limitations and strengthen model validation. This ensured sufficient volume and diversity to train, test, and stress the predictive models.

In the model development phase, several algorithms were evaluated, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Gradient Boosted Trees (GBT). While linear models provided baseline insights and Random Forests improved performance, the Optimized Gradient Boosted Tree model outperformed all others by balancing prediction accuracy, robustness, and interpretability. This model achieves an average deviation of 36.292 units from the test values and explaining 92.9% of the variability in corn yield production. Hyperparameters were tuned with Hyperopt, and MLflow tracked experiments, metrics, and parameters, enabling transparent model comparisons.

The model included an extensive Exploratory Data Analysis (EDA), which addressed missing values, analyzed univariate distributions, and assessed feature importance. Details of the EDA process are available in the [notebook](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/notebook.ipynb).

The prediction application was coded in python using a distribution of Anaconda. Conda was used to manage isolated virtual environments and install all the packages needed without conflicts. This solution was built using Flask, a lightweight and flexible Python web framework, to efficiently handle HTTP requests and deliver a user-friendly interface for interacting with the predictive model. Flask was chosen for its simplicity, scalability, and suitability for developing APIs that serve the predictive service.

To ensure portability and consistent deployment across different environments, the application is containerized using Docker. This approach encapsulates the entire application through a [Dockerfile](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Dockerfile), including the Flask backend, the predictive model, and all dependencies, within a lightweight container. Docker allows the service to run seamlessly across various platforms, eliminating compatibility issues and simplifying deployment.

Together, Flask and Docker provide a robust foundation for the application [predict.py](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/predict.py), enabling efficient development, deployment, and scalability while ensuring reliability and ease of maintenance.

Once the best model was finalized, it was containerized with Docker and pushed to Google Container Registry (GCR). Deployment to Google Cloud Run exposed a REST API endpoint, enabling secure and scalable access for the Mayor’s Office. This serverless design provided elasticity to adapt to fluctuating demand while keeping costs efficient.

Pipeline orchestration was automated using Kestra, deployed on a dedicated Google VM. Kestra ensured reproducibility by automating workflows for data preprocessing, model training, evaluation, deployment, and monitoring. This reduced manual intervention and allowed consistent execution across iterations.

For monitoring and validation, Evidently AI was integrated to track model performance over time. Reports covered training and testing data, data drift detection, and data quality checks. The Evidently UI, deployed in Google Cloud Run, gave stakeholders visibility into model behavior and potential risks.

The entire infrastructure stack—including Cloud Run services, GCS buckets, and networking—was provisioned with Terraform, ensuring reproducibility and version control. To accelerate delivery, GitHub Actions pipelines automated building, testing, and deployment processes. Additionally, integration testing with Docker Compose validated interoperability between MLflow, Evidently, Kestra, and supporting services before deployment.

For artifact and report management, a dedicated structure in Google Cloud Storage separated MLflow artifacts from Evidently reports. This setup ensured governance, traceability of models, and easier long-term maintenance.

![Solution](https://github.com/bizzaccelerator/corn-yield-prediction/blob/main/Images/Solution.JPG)
Photo: Diagram of the prediction service disposed.

### _Key Benefits:_

- Accuracy: The optimized Gradient Boosted Tree model provides high predictive performance, explaining over 90% of corn yield variability.
- Scalability: Cloud Run and Docker ensure seamless scaling with fluctuating demand, while Terraform provisions reproducible infrastructure.
- Transparency & Monitoring: MLflow and Evidently guarantee experiment tracking, model comparisons, drift detection, and visibility into system behavior.
- Automation & Reliability: Kestra orchestrates end-to-end workflows, reducing manual intervention and ensuring repeatable, reliable execution.
- Governance & Maintenance: Dedicated GCS storage separates artifacts and reports, supporting strong governance and easier long-term management.

## How to run the project.

Check out the project [wiki](https://github.com/bizzaccelerator/corn-yield-prediction/wiki) for more details about this repository and step-by-step instructions to reproduce the solution.

## License

This project is no longer open source. It's no longer licensed under MIT.

As of September 19, 2025, all rights are reserved.

- You may not copy, modify, or distribute this code without explicit permission.
- Past versions released under the MIT License remain available under that license.
- Future versions (commits after this date) are proprietary.
