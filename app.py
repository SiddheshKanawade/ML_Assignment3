

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Set the page title
st.set_page_config(page_title="Logistic Regression Demo")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Define the logistic regression function
def logistic_regression(X, y, epochs):
    # Initialize the logistic regression model
    print(epochs)
    model = LogisticRegression(solver='saga', max_iter=epochs)

    # Train the model
    model.fit(X, y)

    # Predict the class labels for the input data
    y_pred = model.predict(X)

    return model, y_pred, model.coef_, model.intercept_

# Define the function to plot the decision boundary
def plot_decision_boundary(X, y, model):
    # Create a meshgrid of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the class labels for the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')

# Define the Streamlit app
def app():
    # Set the app title
    st.title("Logistic Regression Demo")

    # Define the dataset selection dropdown
    dataset = st.selectbox(
        'Select a dataset:',
        ('Iris', 'Breast Cancer', 'Wine', 'Digits')
    )

    # Load the selected dataset
    if dataset == 'Iris':
        X, y = datasets.load_iris(return_X_y=True)
        X = X[:, :2]  # we only take the first two features.
    elif dataset == 'Breast Cancer':
        X, y = datasets.load_breast_cancer(return_X_y=True)
        X = X[:, :2]
    elif dataset == 'Wine':
        X, y = datasets.load_wine(return_X_y=True)
        X = X[:, :2]
    elif dataset == 'Digits':
        X, y = datasets.load_digits(return_X_y=True)
        X = X[:, :2]
    else:
        st.error("Invalid dataset selected.")

    # Define the epochs slider
    epochs = st.slider("Epochs", 0, 1000, 0, 5)

    # Plot the decision boundary for the selected epochs
    if epochs > 0:
        model, y_pred, coef, intercept = logistic_regression(X, y, epochs)
        plt.figure()
        plot_decision_boundary(X, y, model)
        st.pyplot()
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        st.pyplot()


# Run the app
if __name__ == "__main__":
    app()
