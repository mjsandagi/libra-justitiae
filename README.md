# Libra Justitiae

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)

## Overview

**Libra Justitiae** (tl. _Scales of Justice_) is a cloud-native predictive machine learning application that determines prosecution sentences for a given criminal based on specific input parameters. The application is containerised using Docker and can be deployed to Kubernetes clusters for scalable, production-ready deployments.

### Key Features

-   Predicts the type of prosecution for criminals aged 18 or above.
-   Provides a web interface for users to input criminal data and receive sentencing predictions.
-   Containerised architecture using Docker for consistent deployment across environments.
-   Kubernetes-ready with deployment manifests for scalable orchestration.
-   Cloud-native design with support for horizontal scaling and load balancing.

## Development Information

### Front-End

The front-end is built using modern web technologies to ensure a responsive and user-friendly interface.

-   **JavaScript**: Interactive elements and dynamic content are handled using JavaScript, enhancing the user experience.
-   **API Integration**: The front-end communicates with the back-end through RESTful APIs to fetch and submit data from/to a proxy Flask server, ensuring seamless interaction between the user interface and the server.

Users can input criminal data through forms, which are then validated and sent to the back end for processing. The front-end also displays the results of the predictive model, providing users with insights into the type of prosecution for the given criminal data.

### Back-End

The back-end is responsible for processing the criminal data, training the predictive model, and providing the front-end with the necessary information. Here are the key components:

-   **Flask**: A micro web framework used to create a proxy server to accept RESTful APIs and handle incoming requests from the front-end.
-   **Machine Learning (TensorFlow/Keras)**: The predictive model is trained using machine learning algorithms and a neural network to classify the type of prosecution based on the input parameters.
-   **Containerisation**: The application is packaged in a Docker container, ensuring consistency across different deployment environments.
-   **Cloud-Native Architecture**: Designed to run in Kubernetes clusters with support for multiple replicas and load balancing.

### Machine Learning Model

Machine learning is used to predict the type of prosecution for criminals based on their past criminal records. Specifically, the model predicts two key outcomes: **sentences served (imprisonment)** and **fines imposed**. The model is built using **TensorFlow** and **Keras**.

#### 1. **Model Architecture**

The model is a **feedforward neural network**, which consists of the following layers:

-   **Input Layer**: Accepts two input features, namely:
    -   **Main Source of Conviction** (a phrase).
    -   **Past Convictions** (a numerical value).
-   **Hidden Layers**: The network has two hidden layers with **32** and **24** neurons, respectively, each using **ReLU** (Rectified Linear Unit) and **ELU** (Exponential Linear Unit) activation functions.
-   **Output Layer**: A single neuron that outputs the predicted value, either for the **sentences served** or the **amount fined**. These outputs are continuous values.

#### 2. **Data Preprocessing**

The data undergoes several key preprocessing steps:

-   **Label Encoding**: The **Main Source of Conviction** (e.g., theft, assault) is encoded into numerical values using `LabelEncoder` to make it compatible with the neural network.
-   **Feature Scaling**: The **Past Convictions** feature is scaled using **MinMaxScaler** so that all input features lie between 0 and 1, optimising the neural network training process.

#### 3. **Training the Model**

The model is trained using the **Nadam optimiser** and **Mean Absolute Error (MAE)** loss function. The training process involves the following:

-   **Epochs**: The model is trained for 69 epochs to optimise the weights.
-   **Batch Size**: The training uses a batch size of 12 for efficient updates during training.

#### 4. **Predictions**

Once trained, the model is used to predict:

-   **Sentence Length**: The number of years a criminal is likely to serve in prison based on their criminal record.
-   **Amount Fined**: The suggested fine that should be imposed based on the input data.

Additionally, the model determines the **statistically preferred punishment**—whether imprisonment or a fine is more likely based on historical trends in the data.

#### 5. **Output**

The output of the model includes:

-   **Predicted Sentence to be Served**: The expected duration of imprisonment (in years).
-   **Predicted Amount to Fine**: The expected amount to be fined (in pounds sterling).
-   **Statistically Preferred Punishment**: The punishment type that is statistically more likely for a given criminal conviction.

##### Example Output

```JSON
{
    "Predicted Sentence to be Served: ": 9.7,
    "Predicted Amount to Fine: ": 289469.73,
    "Statistically Preferred Punishment": "Fine"
}
```

## Deployment

### Prerequisites

Before deploying the application, ensure you have the following installed:

-   **Docker**: For containerising the application ([Install Docker](https://docs.docker.com/get-docker/))
-   **Kubernetes**: For orchestration (either Docker Desktop with Kubernetes enabled or Minikube)
-   **kubectl**: Kubernetes command-line tool ([Install kubectl](https://kubernetes.io/docs/tasks/tools/))
-   **Minikube** (optional): For local Kubernetes clusters ([Install Minikube](https://minikube.sigs.k8s.io/docs/start/))

### Local Development with Docker

#### Building the Docker Image

To build the Docker image locally:

```bash
docker build -t libra-local:v1 .
```

This command creates a Docker image tagged as `libra-local:v1` containing the Flask application and all its dependencies.

#### Running the Container Locally

To run the application in a Docker container:

```bash
docker run -p 5000:5000 libra-local:v1
```

The application will be accessible at `http://localhost:5000`.

### Kubernetes Deployment with Minikube

#### Starting Minikube

First, start your local Kubernetes cluster:

```bash
minikube start
```

#### Loading the Docker Image into Minikube

Since Minikube runs in its own Docker environment, you need to load the locally built image:

```bash
minikube image load libra-local:v1
```

#### Deploying to Kubernetes

Apply the Kubernetes deployment and service configurations:

```bash
kubectl apply -f deployment.yaml -f service.yaml
```

This creates:

-   **Deployment**: 3 replicas of the application for high availability
-   **Service**: LoadBalancer service exposing the application on port 80

#### Checking Deployment Status

Monitor the rollout status:

```bash
kubectl rollout status deployment/libra-deployment
```

View running pods:

```bash
kubectl get pods
```

#### Accessing the Application

Get the service URL:

```bash
minikube service libra-service --url
```

Open the returned URL in your browser to access the application.

### Automated Deployment Script

For convenience, use the provided deployment script:

```bash
./deploy.sh
```

This script automates the entire deployment process:

1. Builds the Docker image
2. Starts Minikube (if not already running)
3. Loads the image into Minikube
4. Applies Kubernetes configurations
5. Waits for successful deployment
6. Displays the service URL

### Kubernetes Configuration Details

#### Deployment Configuration

The deployment (`deployment.yaml`) specifies:

-   **Replicas**: 3 instances for load distribution
-   **Image Pull Policy**: `Never` (uses local image)
-   **Container Port**: 5000 (Flask default)

#### Service Configuration

The service (`service.yaml`) provides:

-   **Type**: LoadBalancer for external access
-   **Port**: 80 (external) → 5000 (container)
-   **Selector**: Routes traffic to pods labelled `app: libra-app`

### Managing the Deployment

#### Viewing Logs

To view logs from a specific pod:

```bash
kubectl logs <pod-name>
```

To follow logs in real-time:

```bash
kubectl logs -f <pod-name>
```

#### Scaling the Deployment

To scale the number of replicas:

```bash
kubectl scale deployment/libra-deployment --replicas=5
```

#### Updating the Application

After making code changes:

1. Rebuild the Docker image:
    ```bash
    docker build -t libra-local:v1 .
    ```
2. Reload the image into Minikube:
    ```bash
    minikube image load libra-local:v1
    ```
3. Restart the deployment:
    ```bash
    kubectl rollout restart deployment/libra-deployment
    ```

#### Cleaning Up

To delete the deployment and service:

```bash
kubectl delete -f deployment.yaml -f service.yaml
```

To stop Minikube:

```bash
minikube stop
```

## Credits

-   [@mjsandagi](https://github.com/mjsandagi)
-   [@SakuragamaRykii](https://github.com/SakuragamaRykii)
-   [@Akhilesh271](https://github.com/Akhilesh271)
