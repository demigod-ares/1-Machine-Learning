#PCA to Speed-up Machine Learning Algorithms

#Download and Load the Data
from sklearn.datasets import fetch_openml
mnist = fetch_openml('MNIST original')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)