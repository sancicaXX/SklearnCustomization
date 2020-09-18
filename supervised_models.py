# -*- coding: utf-8 -*-
# @Time: 2020/9/17 15:27
# @Author: sancica
# @File: supervised_models.py
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


def sgd_classifier(data, labels, args):
    """
    随机梯度下降分类器
    """
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(**args)
    clf.fit(data, labels)
    return clf


def sgd_regression(data, labels, args):
    """
    随机梯度下降回归
    """
    from sklearn.linear_model import SGDRegressor
    reg = SGDRegressor(**args)
    reg.fit(data, labels)
    return reg


def nearest_neighbors(data, args):
    """
    最近邻
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(**args)
    nbrs.fit(data)
    # 计算测试数据对应的最近邻下标和距离
    # distances, indices = nbrs.kneighbors(test_data)
    return nbrs


def gaussian_NB(data, labels, args={}):
    """
    高斯贝叶斯
    """
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB(**args)
    gnb.fit(data, labels)
    return gnb


def decision_tree_classifier(data, labels, args={}):
    """
    决策树分类器
    """
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(**args)
    clf.fit(data, labels)
    return clf


def decision_tree_regression(data, labels, args={}):
    """
    决策树回归
    """
    from sklearn.tree import DecisionTreeRegressor
    reg = DecisionTreeRegressor(**args)
    reg.fit(data, labels)


def bagging_classifier(clf, args={}):
    """
    bagging集成模型
    """
    from sklearn.ensemble import BaggingClassifier
    bagging = BaggingClassifier(clf, **args)
    return bagging


def random_forest(data, labels, args={}):
    """
    随机森林，一种bagging集成模型
    """
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(**args)
    clf.fit(data, labels)
    return clf


def adaboost_classifier(data, labels, args={}):
    """
    boosting算法：AdaBoost分类
    """
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(**args)
    clf.fit(data, labels)
    return clf


def adaboost_regressor(data, labels, args={}):
    """
    boosting算法：Adaboost回归
    """
    from sklearn.ensemble import AdaBoostRegressor
    reg = AdaBoostRegressor(**args)
    reg.fit(data, labels)
    return reg


def GBDT_classifier(data, labels, args={}):
    """
    boosting算法：GBDT分类
    """
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(**args)
    clf.fit(data, labels)
    return clf


def GBDT_regressor(data, labels, args={}):
    """
    boosting算法：GBDT回归
    """
    from sklearn.ensemble import GradientBoostingRegressor
    reg = GradientBoostingRegressor(**args)
    reg.fit(data, labels)
    return reg


def stacking_classifier(estimators, final_estimator, data, labels, args={}):
    """
    Stacking算法：通过多个模型降低bias, 分类
    """
    from sklearn.ensemble import StackingClassifier
    clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, **args)
    clf.fit(data, labels)
    return clf


def stacking_regressor(estimators, final_estimator, data, labels, args={}):
    """
    Stacking算法：通过多个模型降低bias， 回归
    """
    from sklearn.ensemble import StackingRegressor
    reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator, **args)
    reg.fit(data, labels)
    return reg




if __name__ == "__main__":
    model = decision_tree_classifier([[1, 1], [2, 3], [3, 4]], [1, 2, 3])
    from plot_imgs import dt_structure

    dt_structure(model, "pic")
    # print(model.predict([[1, 1]]))
    # print(model.coef_)
