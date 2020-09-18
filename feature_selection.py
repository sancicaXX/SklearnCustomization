# -*- coding: utf-8 -*-
# @Time: 2020/9/18 9:50
# @Author: sancica
# @File: feature_selection.py
# functional description: feature selection method

def variance_threshold(data, threshold):
    """
    基于方差阈值进行特征选择
    """
    from sklearn.feature_selection import VarianceThreshold
    return VarianceThreshold(threshold=threshold).fit_transform(data)


def select_K_best(data, labels, selector, k):
    """
    通过单变量评分进行特征选择
    selector：单变量评分方法
    """
    from sklearn.feature_selection import SelectKBest
    return SelectKBest(selector, k=k).fit_transform(data, labels)


def L1(data, labels, args={}):
    """
    通过模型进行特征选择，这里选取L1权重为非0的特征
    """
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel
    lsvc = LinearSVC(**args).fit(data, labels)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(data)


def extra_tree(data, labels, args={}):
    """
    通过模型进行特征选择，这里选择extra_tree选择特征
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(**args).fit(data, labels)
    model = SelectFromModel(clf, prefit=True)
    return model.transform(data)


