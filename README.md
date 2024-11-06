# Boston Housing Price Prediction

## Overview

This web application predicts the **median home values** in Boston based on various features such as **crime rate**, **number of rooms**, **property-tax rate**, and other socio-economic factors. It uses a **Linear Regression** model trained on the well-known **Boston Housing dataset**.

The app allows users to input different features to predict the housing price for a specific area, giving valuable insights into real estate trends in Boston. The model evaluates the input data and returns the predicted price, making it an excellent tool for real estate professionals, homebuyers, or anyone interested in understanding housing prices based on different factors.

---

## Features

- **Interactive User Interface**: The app provides sliders and input fields to adjust the features.
- **Prediction of Home Price**: Enter your features, click the button to get the predicted home price.
- **Model Performance Metrics**: The app displays the evaluation metrics (Mean Absolute Error, Mean Squared Error, and Root Mean Squared Error) to give insights into the model's accuracy.

---

## How It Works

### Data Preprocessing
The app uses the **Boston Housing dataset**. The features are:
- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for large lots
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxide concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built before 1940
- **DIS**: Weighted distance to employment centers
- **RAD**: Accessibility to radial highways
- **TAX**: Property tax rate
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: Proportion of black residents by town
- **LSTAT**: Percentage of lower status population

### Model
- **Linear Regression** is used to model the relationship between these features and the target variable (`medv`), which represents the **median home value**.

