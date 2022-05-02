import numpy as np
import torch.nn as nn
import torch
from math import pi, exp, sqrt
import matplotlib.pyplot as plt

#PyTorch module describing an ARIMA(0,1,1) time series
class ARIMA011(nn.Module):
    """
    ARIMA(0,1,1) is fact MA(1) on the differentiations. Thus we will initiate the class with the
    MA(1) on the differentiations - X(t) = drift + theta * e(t-1) + e(t) where:
    drift is the interceptor
    theta is the coefficient of the previous error
    e(t-1) is the previous error (err_last)
    e(t) is the current error - we don't use it because this is the error between the real value and the predicted
    """
    def __init__(self):
        super(ARIMA011, self).__init__()
        self.drift = nn.parameter.Parameter(torch.rand(1))
        self.theta = nn.parameter.Parameter(torch.rand(1))

    def forward(self, err_last):
        return self.drift + self.theta * err_last


def generate_arima011_samples(drift_generated, theta_generated, sigma, sample_size):
    """
    Random time series generation with ARIMA(0,1,1)
    generating 'sample_size' samples of ARIMA(0,1,1) with random drift and theta
    and e(t)~N(0,sigma^2). we will use MA(1) on the differentiations - Z(t) = drift + theta * e(t-1) + e(t)
    and then we will use the formula X(t) = X(0) + (z(1)+...z(t)) where X(0) is 0 in this case because we are
    generating a new series. note that this time we do add e(t) because we generate a serious and not a prediction
    """
    err = torch.randn(sample_size)*sigma
    err_last = torch.tensor([0 if i == 0 else err[i - 1] for i in range(len(err))])
    # e(t-1) was drifted so it's indexes will match the indexes of e(t)

    z_diff = drift_generated + theta_generated * err_last + err
    x_generated = z_diff.cumsum(dim=0)
    # cumulative sum of the differentiations

    return x_generated

def predict(x_set, model):
    """
    Prediction function
    Inputs:
    x_set - samples of generated series
    model - the current hypothesis of learning
    Outputs:
    pred_diff - prediction on the differentiation series
    pred_x - prediction on the original series based on pred_diff
    """

    x_diff = x_set[1:]-x_set[:-1]
    e_prev = torch.zeros(1)
    pred_diff = torch.zeros_like(x_diff)
    pred_x = torch.zeros_like(x_set)+x_set[0]
    # defining the differentiations series and initializing the last_error to 0 before iterating
    # initializing pred_diff with 0 and pred_x with x(0) - this time not 0 cause it an existing value

    for i, diff in enumerate(x_diff):
        pred_diff[i] = model(e_prev)
        pred_x[i+1:] += pred_diff[i]
        e_prev = diff - pred_diff[i]
    # updating the hypothesis based on the last error
    # accumulating the rolling sum of the predicted differentiations
    # updating the current error to be the last for the next iteration

    return pred_diff, pred_x


def model_fit(x_train, model, epochs, learning_rate):
    """
    model fitting function
    Inputs:
    x_train - samples of the train from the generated series
    model - the current hypothesis of learning - being updated every iteration with the learning rate
    epochs - number of epochs of the model
    learning rate - learning rate of the model
    Outputs:
    pred_x - predicted series after executing training
    """

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    """
    we are fitting the model according to maximum likelihood estimator. Thus maximize formula:
    - prod of exp( - (x_train[i] - pred_x[i])**2 / (2 * sigma**2)) / sqrt(2 * pi * sigma**2).
    if we take log of the formula we get that we need to maximize the expression  - (x_train[i] - pred_x[i])**2
    which is like minimizing (x_train[i] - pred_x[i])**2 so we can look at this problem as MSE minimization where
    y is x_train, x is the last error(e(t-1)) and the hypothesis is pred_x.
    
    we used for that in MSELoss with Adam optimizer(we could use others also)
    """

    for epoch in range(epochs):
        pred_diff, pred_x = predict(x_train, model)
        x_diff = x_train[1:]-x_train[:-1]
        loss = criterion(pred_diff, x_diff)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # for each epoch - we predict the current predicted for the current hypothesis
    # calculating the loss
    # initializing grads back to zero
    # calculating the backwards derivatives
    # executing the optimization

    return pred_x


def probability(x_test, model, sigma, test_size):
    """
    probability calculating
    Inputs:
    x_test - samples of the test from the generated series
    model - the fitted model
    sigma
    Outputs:
    probability
    """
    pred_diff_test, pred_x_test = predict(x_test, model)
    pred_x_test = pred_x_test[-test_size:]
    prod = 1
    for i in range(test_size):
        prod *= exp( - (x_test[-test_size+i] - pred_x_test[i])**2 / (2 * sigma**2)) / sqrt(2 * pi * sigma**2)
    # calculating the prediction of the model on the test set
    # calculating the probability according to the formula previously mentioned.
    return prod


def main():
    drift_generated = np.random.rand()
    theta_generated = np.random.rand()
    sigma = 1
    sample_size = 100
    x_generated = generate_arima011_samples(drift_generated, theta_generated, sigma, sample_size)
    plt.plot(range(sample_size), x_generated)
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.show()
    # initiating parameters for generating the ARIMA(0,1,1) series
    # plotting the generated series

    train_size = 80
    x_train = x_generated[:train_size]
    model = ARIMA011()
    pred_x = model_fit(x_train, model, 100, 0.1)
    plt.plot(range(train_size), x_train)
    plt.plot(range(train_size), pred_x.detach())
    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.legend(['generated series', 'predicted series'])
    plt.show()
    # initializing first hypothesis with the module of the class we built and then fitting the model
    # with 100 epochs and learning rate 0.1
    # plotting the generated series and the predicted series

    test_size = sample_size - train_size
    x_test = x_generated
    print(probability(x_test, model, sigma, test_size))
    # calculating the probability of observing testing data set
    # we take all the data for predicting and the we will slice only the test.

    '''
    for Task 1.2.2 i will take the samples 1-7 and 14-20 and fit an ARIMA(0,1,1) model which maximize
    the likelihood. the accuracy will decrease a little bit, because there are 2 different initials instead
    of one (on the first and on the 14th sample). then we will apply the model on 8 based on the error of 7
    and so on.
    '''

    plt.plot(np.random.rand(10))
    plt.show()

if __name__ == '__main__':
    main()
