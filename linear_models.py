# -*- coding: utf-8 -*-
# @Time: 2020/9/17 15:27
# @Author: sancica
# @File: linear_models.py
# functional description: linear models for sklearn


def linear_regression(data, labels, args):
    """
    线性回归模型
    """
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(**args)
    reg.fit(data, labels)
    return reg


def ridge(data, labels, alpha, args):
    """
    岭回归
    alpha: 惩罚因子，越大越能减轻多重共线性的影响
    """
    from sklearn.linear_model import Ridge
    reg = Ridge(alpha, **args)
    reg.fit(data, labels)
    return reg


def ridgeCV(data, labels, alphas, args):
    """
    使用交叉验证的岭回归，使用交叉验证获取最优的alpha值
    alphas：提供的惩罚因子列表
    """
    from sklearn.linear_model import RidgeCV
    reg = RidgeCV(alphas=alphas, **args)
    reg.fit(data, labels)
    return reg


def lasso(data, labels, alpha, args):
    """
    lasso回归
    alpha:惩罚因子
    """
    from sklearn.linear_model import Lasso
    reg = Lasso(alpha=alpha, **args)
    reg.fit(data, labels)
    return reg


def lassoCV(data, labels, alphas, args):
    from sklearn.linear_model import LassoCV
    reg = LassoCV(alphas=alphas, **args)
    reg.fit(data, labels)
    return reg


def bayesian_ridge(data, labels, args):
    """
    采用bayes岭回归
    """
    from sklearn.linear_model import BayesianRidge
    reg = BayesianRidge(**args)
    reg.fit(data, labels)
    return reg


def logistic(data, labels, args):
    """
    逻辑回归
    """
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(**args)
    reg.fit(data, labels)
    return reg


def svmR(data, labels, args):
    """
    svm进行回归
    """
    from sklearn.svm import SVR
    reg = SVR(**args)
    reg.fit(data, labels)
    return reg


def svmC(data, labels, args):
    """
    svm进行分类
    """
    from sklearn.svm import SVC
    clf = SVC(**args)
    clf.fit(data, labels)
    return clf


if __name__ == "__main__":
    model = linear_regression([[1, 1], [2, 3], [3, 4]], [1, 2, 3], {"fit_intercept": False})
    print(model.predict([[1, 1]]))
    print(model.coef_)
