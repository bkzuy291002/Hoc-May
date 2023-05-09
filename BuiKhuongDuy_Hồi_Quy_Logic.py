#Bùi Khương Duy 20000535
#Ví Dụ 1
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# extended data by adding a column of 1s (x_0 = 1)
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)
plt.show()
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    # method to calculate model logistic regression by Stochastic Gradient Descent method
    # eta: learning rate; tol: tolerance; max_count: maximum iterates
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    
    # loop of stochastic gradient descent
    while count < max_count:
        # shuffle the order of data (for stochastic gradient descent).
        # and put into mix_id
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            
            # stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
print(sigmoid(np.dot(w[-1].T, X)))
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

xx = np.linspace(0, 6, 1000)
w0 = w[-1][0]
w1 = w[-1][1]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
x_test = np.array([2.45, 1.85, 3.75, 3.21, 4.05])
y_pred = np.array([1 if x > threshold else 0 for x in x_test])
print(y_pred)
#Ví Dụ 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# generate list of data points
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X1 = np.random.multivariate_normal(means[0], cov, N)
X2 = np.random.multivariate_normal(means[1], cov, N)
plt.plot(X1[:, 0], X1[:, 1], 'bs', markersize = 8, alpha = 1)
plt.plot(X2[:, 0], X2[:, 1], 'ro', markersize = 8, alpha = 1)
plt.axis('equal')
plt.ylim(0, 4)
plt.xlim(0, 5)

# hide tikcs
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.xlabel('$x_1$', fontsize = 20)
plt.ylabel('$x_2$', fontsize = 20)
plt.show()
def sigmoid(s):
    return 1/(1 + np.exp(-s)) # calculate sigmoid function

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    lambda_ = 0.0001
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data for stochastic gradient descent method
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi - lambda_*w[-1]
            count += 1
            # stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
X = np.concatenate((X1, X2), axis = 0).T
y = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis = 1).T

# Xbar
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

eta = 0.05
d = X.shape[0]
w_init = np.random.randn(d, 1) # initialize parameters w = w_init

# call logistic_sigmoid_regression procedure
w = logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count= 10000)

# print out the parameter
print(w[-1])
# Make data.
x1m = np.arange(-1, 6, 0.025) # generate data coord. X1
xlen = len(x1m)
x2m = np.arange(0, 4, 0.025) # generate data coord. X2
x2en = len(x2m)
x1m, x2m = np.meshgrid(x1m, x2m) # create mesh grid X = (X1, X2)

# now assign the parameter w0, w1, w2 from array w which was computed above
w0 = w[-1][0][0]
w1 = w[-1][1][0]
w2 = w[-1][2][0]

# calculate probability zm=P(c|x)=sigmoid(w^Tx)=sigmoid(w0+w1*x1m+w2*x2m)
zm = sigmoid(w0 + w1*x1m + w2*x2m)

# plot contour of prob. zm by the saturation of blue and red
# more red <=> prob. that data point belong to red class is higher & vise versa
CS = plt.contourf(x1m, x2m, zm, 200, cmap='jet')

# finally, plot the data and take a look
plt.plot(X1[:, 0], X1[:, 1], 'bs', markersize = 8, alpha = 1)
plt.plot(X2[:, 0], X2[:, 1], 'ro', markersize = 8, alpha = 1)
plt.axis('equal')
plt.ylim(0, 4)
plt.xlim(0, 5)

# hide tikcs
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.xlabel('$x_1$', fontsize = 20)
plt.ylabel('$x_2$', fontsize = 20)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

X = np.concatenate((X1, X2), axis = 0).T
y = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis = 1).T

logreg = LogisticRegression()
logreg.fit(X.T, y.ravel())
print('X = ', X)
print('Intercept: ', logreg.intercept_)
print('Coefficients: ', logreg.coef_)
print('Accuracy: ', logreg.score(X.T, y.ravel()))
print('Confusion matrix: ', metrics.confusion_matrix(y.ravel(), logreg.predict(X.T)))
#Ví Dụ 3
# importing module
import numpy as np
import pandas as pd

data = pd.read_csv("C:\Python\Học Máy\Admission_Predict.csv")
data
# Split data into X and y
X = data.iloc[:, 1:8]
X
y = data.iloc[:, 8]
y
y = data.iloc[:, 8]
y
# Split data into training and test sets
X_train = X[:350]
X_test = X[350:]
y_train = y[:350]
y_test = y[350:]
#a Phân loại bằng phương pháp hồi quy Logistic
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time

# Create and train a logistic regression model
logreg = LogisticRegression(max_iter = 10000)
y_train_classified = np.where(y_train >= 0.75, 1, 0)
start_time = time.time()
logreg.fit(X_train, y_train_classified)
end_time = time.time()
print("Training time: ", end_time - start_time)

# Print the intercept and coefficients
print("Intercept: ", logreg.intercept_)
print("Coefficients:\n", logreg.coef_)

# Predict the response for test dataset
y_pred = logreg.predict(X_test)

# Calculate the accuracy, precision and recall
y_test_classified = np.where(y_test >= 0.75, 1, 0)
print("Accuracy:  ", metrics.accuracy_score(y_test_classified, y_pred))
print("Precision: ", metrics.precision_score(y_test_classified, y_pred))
print("Recall:    ", metrics.recall_score(y_test_classified, y_pred))
#b Dự đoán khả năng bằng hồi quy tuyến tính
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Create and train the model
linreg = LinearRegression()

# Train the model using the training sets
linreg.fit(X_train, y_train)

# Print the intercept and coefficients
print('Intercept: ', linreg.intercept_)
print('Coefficients: ', linreg.coef_)

# Make predictions using the testing set
y_pred = linreg.predict(X_test)

# Mean Squared Error
print('MSE: ', metrics.mean_squared_error(y_test, y_pred))
#c Sử dụng phương pháp Naïve Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import time

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
y_train_classified = np.where(y_train >= 0.75, 1, 0)
start_time = time.time()
model.fit(X_train, y_train_classified)
end_time = time.time()
print(f'Training time: {end_time - start_time} seconds')

# Predict response for test dataset
y_pred = model.predict(X_test)

# Model Evaluation
y_test_classified = np.where(y_test >= 0.75, 1, 0)
print("Accuracy:", metrics.accuracy_score(y_test_classified, y_pred))
print("Precision:", metrics.precision_score(y_test_classified, y_pred))
print("Recall:", metrics.recall_score(y_test_classified, y_pred))