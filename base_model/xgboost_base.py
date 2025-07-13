from xgboost.sklearn import XGBClassifier
# 模型初始化
def get_xgboost():
    clf = XGBClassifier(
        booster = 'gbtree',
        objective = 'multi:softmax',
        num_class = 7,
        gamma = 0.1, # 最小分裂损失
        max_depth = 6, # 树的最大深度
        reg_lambda = 2, # L2正则化
        subsample = 1, # 子采样率
        colsample_bytree = 0.8, # 每棵树随机采样的列数占比
        min_child_weight = 1, # 最小叶子节点样本权重和
        eta = 0.3, # 学习率
        seed = 1000, # 随机种子
        nthread = 4, # 线程数
        n_estimators = 500 # 迭代次数
    )
    return clf