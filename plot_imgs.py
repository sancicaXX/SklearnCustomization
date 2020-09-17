# -*- coding: utf-8 -*-
# @Time: 2020/9/17 18:47
# @Author: sancica
# @File: plot_imgs.py
# functional description: plot images including tree structure, scatter, pie, plot, matplotlib, seaborn and so on.


def dt_structure(clf, file_name, args={}):
    from sklearn import tree
    import pydotplus
    dot_data = tree.export_graphviz(clf, **args)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./result/"+file_name+".pdf")
    print("store decision tree structure to "+file_name+".pdf")


