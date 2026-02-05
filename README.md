# Kaggle Competition House Price Prediction

This repository contains a Deep Learning approach to the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The goal is to predict the final price of residential homes in Ames, Iowa, using 79 explanatory variables.

## 🚀 Overview
The solution utilizes a **Deep Neural Network (DNN)** built with **TensorFlow/Keras**. It includes a robust preprocessing pipeline to handle the mix of categorical and numerical data common in real estate datasets.

## 🛠️ Technical Workflow
* **Data Cleaning**: Automated handling of missing values. Numerical nulls are filled with `-1`, and categorical nulls are tagged as `"NaN"`.
* **Feature Encoding**: Categorical variables are transformed using `LabelEncoder` to prepare them for the neural network.
* **Normalization**: A `tf.keras.layers.Normalization` layer is adapted to the training data to ensure all features are on a similar scale, improving convergence.
* **Model Architecture**: 
    * Input Normalization Layer.
    * Two Dense hidden layers (64 units each) with ReLU activation.
    * Single linear output unit for regression.
* **Training**: Optimized using the **Adam** optimizer and **Mean Absolute Error (MAE)** loss function.

## 📁 Project Structure
* `Model.ipynb`: The primary Google Colab notebook containing data loading, EDA, preprocessing, model training, and prediction logic.
* `train.csv / test.csv`: Competition datasets (not included; download from Kaggle).
* `submission.csv`: The final predicted sales prices for the test set.

## 📈 Results
The model's training performance is visualized via loss curves (MAE over epochs) within the notebook to monitor for overfitting and convergence.
