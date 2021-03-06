# -*- coding: utf-8 -*-
# @Time: 2020/9/21 15:56
# @Author: sancica
# @File: utils.py
# functional description: utils


# ------------------------------------------------异常值检测------------------------------------------
def isolation_forest(X, args={}):
    """
    孤立森林，用于检测异常值
    """
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(**args)
    clf.fit(X)


def local_outlier_factor(X, args={}):
    """
    局部异常因子：用于检测异常值和novelty detection，如果需要进行novelty detection，需要给定属性novelty=True
    """
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(**args)
    clf.fit(X)
    return clf


# -----------------------------------------------缺失值填充--------------------------------------------
def simple_imputer(X, args={}):
    """
    缺失值插入：使用固定值、均值、中位数、众数
    """
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(**args)
    imp.fit(X)
    return imp


def iterative_imputer(X, args={}):
    """
    缺失值插入：通过将该缺失属性与其他属性结合起来进行插值
    """
    from sklearn.impute import IterativeImputer
    iti = IterativeImputer(**args)
    iti.fit(X)
    return iti


def knn_imputer(X, args={}):
    """
    KNN插值法
    """
    from sklearn.impute import KNNImputer
    imp = KNNImputer(**args)
    imp.fit(X)
    return imp


# ---------------------------------------------概率分布检测----------------------------------------------
def kernel_density(X, args={}):
    """
    密度估计：用于检测数据分布情况
    """
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(**args)
    kde.fit(X)
    return kde


# ---------------------------------------------调整超参数-------------------------------------------------
def grid_search(clf, param_grid, X, y, args={}):
    """
    网格搜索：用于超参数选择
    """
    from sklearn.model_selection import GridSearchCV
    search = GridSearchCV(clf, param_grid, **args)
    search.fit(X, y)
    return search


def validation_curve(clf, X, y, param_name, param_range, args={}):
    """
    使用验证曲线进行调参
    """
    from sklearn.model_selection import validation_curve
    train_scores, valid_scores = validation_curve(clf, X, y, param_name, param_range, **args)
    return train_scores, valid_scores


def learning_curve(clf, X, y, train_sizes, cv, args={}):
    """
    用于判断是否需要更多的样本，样本的多少与模型好坏的判断
    """
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes, cv, **args)
    return train_sizes, train_scores, valid_scores


# ----------------------------------------------数据持久化------------------------------
def dump(obj, name, args={}):
    """
    使用joblib将对象保存
    """
    import joblib
    joblib.dump(obj, name, **args)


def load(name, args={}):
    """
    使用joblib将对象读取
    """
    import joblib
    obj = joblib.load(name, **args)
    return obj
