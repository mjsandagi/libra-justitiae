# Libra Justitiae


![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)


## Overview

**Libra Justitiae** (tl. _Scales of Justice_) is a predictive machine learning model that determines prosecution sentences for a given criminal based on specific input parameters.

### Key Features

-   Predicts the type of prosecution for criminals aged 18 or above.
-   Provides a website interface for users to input criminal data and receive sentencing predictions.

## Development Information

### Front-End

The front-end is built using modern web technologies to ensure a responsive and user-friendly interface.

-   **JavaScript**: Interactive elements and dynamic content are handled using JavaScript, enhancing the user experience.
-   **API Integration**: The front-end communicates with the back-end through RESTful APIs to fetch and submit data from/to a proxy `Flask` server, ensuring seamless interaction between the user interface and the server.

Users can input criminal data through forms, which are then validated and sent to the back end for processing. The front-end also displays the results of the predictive model, providing users with insights into the type of prosecution for the given criminal data.

### Back-End

The back-end is responsible for processing the criminal data, training the predictive model, and providing the front-end with the necessary information. Here are the key components:

-   **Flask**: A micro web framework used to create a proxy server of sorts to accept RESTful APIs and handle incoming requests from the front-end.
-   **Machine Learning (TensorFlow/Keras)**: The predictive model is trained using machine learning algorithms and a neural network to classify the type of prosecution based on the input parameters.

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
"Prediction": {
        "Predicted Sentence to be Served: ": 5.553350746631623,
        "Predicted Amount to Fine: ": 238541.1928635538,
        "Statistically Preferred Punishment": "Fine"
    }
```

## Credits:

-   [@mjsandagi](https://github.com/mjsandagi)
-   [@SakuragamaRykii](https://github.com/SakuragamaRykii)
-   [@Akhilesh271](https://github.com/Akhilesh271)
