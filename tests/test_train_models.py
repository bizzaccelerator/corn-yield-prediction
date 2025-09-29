import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from support_scripts import train_linear, train_ridge, train_lasso, train_gbr


def _synthetic_data():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)
    return X, y


def test_train_linear():
    X, y = _synthetic_data()
    model = train_linear.train(X, y)
    assert isinstance(model, LinearRegression)
    assert model.predict(X).shape == (20,)


def test_train_ridge():
    X, y = _synthetic_data()
    model = train_ridge.train(X, y)
    assert isinstance(model, Ridge)
    assert model.predict(X).shape == (20,)


def test_train_lasso():
    X, y = _synthetic_data()
    model = train_lasso.train(X, y)
    assert isinstance(model, Lasso)
    assert model.predict(X).shape == (20,)


def test_train_gbr():
    X, y = _synthetic_data()
    model = train_gbr.train(X, y)
    assert isinstance(model, GradientBoostingRegressor)
    assert model.predict(X).shape == (20,)
