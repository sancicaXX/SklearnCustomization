# -*- coding: utf-8 -*-
# @Time: 2020/9/17 18:47
# @Author: sancica
# @File: plot_imgs.py
# functional description: plot images including tree structure, scatter, pie, plot, matplotlib, seaborn and so on.


def dt_structure(clf, file_name, args={}):
    """
    决策树结构图
    """
    from sklearn import tree
    import pydotplus
    dot_data = tree.export_graphviz(clf, **args)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./result/" + file_name + ".pdf")
    print("store decision tree structure to " + file_name + ".pdf")


def pdp(clf, X, features, args={}):
    """
    返回pdp的数据，然后自行画图
    """
    from sklearn.inspection import partial_dependence
    pdp, axes = partial_dependence(clf, X, features, **args)


def plot_pdp(clf, X, features, args={}):
    """
    直接画出部分依赖图
    """
    from sklearn.inspection import plot_partial_dependence
    plot_partial_dependence(clf, X, features, **args)
