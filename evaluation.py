# -*- coding: utf-8 -*-
# @Time: 2020/9/18 15:08
# @Author: sancica
# @File: evaluation.py.py
# functional description: evaluation for models including supervised and unsurpervised

# --------------------------------------------------------聚类外在评估方法----------------------------------------------
def rand_index(labels_true, labels_pred):
    """
    聚类评估：兰德系数，外在方法（指用于原始数据的真实标签）
    return: score,float, 最大值为1
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_true, labels_pred)


def mutual_info(labels_true, labels_pred):
    """
    聚类评估：互信息，外在方法
    return：score, float, 最大值为1
    """
    from sklearn.metrics import adjusted_mutual_info_score
    return adjusted_mutual_info_score(labels_true, labels_pred)


def fowlkes_mallows(labels_true, labels_pred):
    """
    聚类评估：TODO:
    """
    from sklearn.metrics import fowlkes_mallows_score
    return fowlkes_mallows_score(labels_true, labels_pred)


# ----------------------------------------------- 聚类内在评估方法-------------------------------------------------
def silhouette(X, labels, args={}):
    """
    聚类评估：轮廓系数，内在方法，没有真实标签
    return:float, -1--1
    """
    from sklearn.metrics import silhouette_score
    return silhouette_score(X, labels, **args)


def calinski_harabasz(X, labels, args={}):
    """
    聚类评估：CHI， 内在方法
    return: score, 分数越大越好
    """
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(X, labels, **args)


def davies_bouldin(X, labels, args={}):
    """
    聚类评估：DBI， 内在方法
    return:score,越接近0越好
    """
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, labels, **args)


def cross_validate(clf, X, y, args={}):
    """
    交叉验证：
    return: accuracy的list
    """
    from sklearn.model_selection import cross_val_score
    return cross_val_score(clf, X, y, **args)


# --------------------------------------------分类评估方法----------------------------------------
def accuracy(y_true, y_pred):
    """
    准确度
    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


def confusion_matrix(y_true, y_pred):
    """
    混淆矩阵, 如果需要打印成图像，则需要import plot_confusion_matrix,然后输入plt.show()
    """
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)


def classification_report(y_true, y_pred, target_names):
    """
    将每一个类的评估信息打印出来
    """
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, target_names)


def precision(y_true, y_pred):
    """
    精度
    """
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    """
    recall
    """
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred)


def F1(y_true, y_pred):
    """
    f1-score
    """
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)


def roc_auc_score(y_true, y_pred):
    """
    roc：如果需要打印出roc曲线的话需要import roc_curve
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)


# -------------------------------------回归评估方法--------------------------------------------

def MSE(y_true, y_pred):
    """
    均方误差
    """
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)


def MAE(y_true, y_pred):
    """
    平均绝对误差
    """
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def MSLE(y_true, y_pred):
    """
    均方对数误差
    """
    from sklearn.metrics import mean_squared_log_error
    return mean_squared_log_error(y_true, y_pred)


def r2_score(y_true, y_pred):
    """
    R方值：该指标能够处理不同量纲下的数据进行评估
    """
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

