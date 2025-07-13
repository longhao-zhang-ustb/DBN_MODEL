from sklearn.datasets import load_breast_cancer
# from genetic_selection import GeneticSelectionCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
# from sklearn.impute import KNNImputer
# from imblearn.over_sampling import BorderlineSMOTE, ADASYN, KMeansSMOTE, SVMSMOTE, SMOTE
from base_model.xgboost_base import get_xgboost
import lightgbm as lgb
from lightgbm import LGBMClassifier
from base_model.lightgbm_base import get_lgb_param
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#############Our Model新增模块#############
import argparse
import os
import utils.tool as tool
import torch
from model_pth.DSSM_Model import DSSMClassifier
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_preprocess.data_loader import prepare_datasets
from model_pth.Fitness import train_and_evaluate
from model_pth.evaluate import evaluate
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
# from imblearn.combine import SMOTETomek, SMOTEENN
# from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, NeighbourhoodCleaningRule, CondensedNearestNeighbour, OneSidedSelection
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import shap
# from catboost import CatBoostClassifier
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder,KBinsDiscretizer
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 设置全局字体大小
plt.rcParams['font.size'] = 20  # 默认字体大小
plt.rcParams['axes.labelsize'] = 20  # 轴标签的字体大小
plt.rcParams['axes.titlesize'] = 20  # 标题的字体大小
plt.rcParams['legend.fontsize'] = 20  # 图例的字体大小

# https://github.com/sdv-dev/SDV
# https://www.modb.pro/db/1702270742289272832

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def data_visual(X, y):
    # 基于tsne进行数据可视化
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_min, X_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - X_min) / (X_max - X_min)
    X_tsne = X_norm
    # 设置绘图样式，包括完全去掉边框
    sns.set_theme(style='white', context='notebook', rc={'axes.edgecolor': 'none', 'axes.linewidth': 0})
    plt.figure(figsize=(6, 6), dpi=160, edgecolor='none') 
    ax = plt.axes(projection='3d')
    for i in range(0, 3):
        ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], X_tsne[y == i, 2], label=f'Class {i}')
    ax.set_xlabel('t-SNE Dimension-1')
    ax.set_ylabel('t-SNE Dimension-2')
    ax.set_zlabel('t-SNE Dimension-3')
    legend = ax.legend(fontsize=8, loc='upper left',  frameon=True, fancybox=True)
    frame = legend.get_frame()
    frame.set_edgecolor('white')
    plt.gca().set_aspect('equal', 'datalim')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 忽略 DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ############################EEG数据降维############################
    ###2处修改内容.....
    df = pd.read_csv(r"base_data\EEG.machinelearing_data_BRMH_6.0.csv")
    model_name = 'best.pth.tar'
    columns_name = 'COH.A.delta.a.FP1.e.Fz'
    shap_index = 1
    # category = 0
    # 6个类别的时候这个地方的代码需要打开
    column = 'Unnamed: 122'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    df = df.drop(columns=[column])
    ############################数据预处理##############################
    # 6大类+1HC
    df_x = df.iloc[:, 1:-1]
    df_y = df.iloc[:, -1]
    y = df.values[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = y.astype(np.float32)
    # 特征标准化
    scaler = MinMaxScaler()
    df_x = pd.DataFrame(scaler.fit_transform(df_x), columns=df_x.columns)
    X = df_x.values
    # data_visual(X, y)
    # exit()
    ###########将数据集划分为多个二分类数据############
    # # 6种类别的映射结果
    # # {'Addictive disorder': 0, 'Anxiety disorder': 1, 
    # # 'Healthy control': 2, 'Mood disorder': 3, 
    # # 'Obsessive compulsive disorder': 4, 'Schizophrenia': 5, 
    # # 'Trauma and stress related disorder': 6}
    # # 12种类别的映射结果
    # # {'Acute stress disorder': 0, 'Adjustment disorder': 1, 
    # # 'Alcohol use disorder': 2, 'Behavioral addiction disorder': 3, 
    # # 'Bipolar disorder': 4, 'Depressive disorder': 5, 'Healthy control': 6, 
    # # 'Obsessive compulsitve disorder': 7, 'Panic disorder': 8, 
    # # 'Posttraumatic stress disorder': 9, 'Schizophrenia': 10, 
    # # 'Social anxiety disorder': 11}
    # df = pd.read_csv(r"base_data\EEG.machinelearing_data_BRMH.csv")
    # column = 'Unnamed: 122'
    # df = df.drop(columns=[column])
    # df_x = df.iloc[:, 8:]
    # df_y = df.iloc[:, 7]
    # y = df_y.values
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    # y = y.astype(np.float32)
    # df_n = pd.concat([df_x, df_y], axis=1)   
    # df_n['specific.disorder'] = y 
    # # label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # unique_values, counts = np.unique(y, return_counts=True)
    # for value in unique_values:
    #     if value != 6.0:
    #         temp_target_ill = df_n[df_n['specific.disorder'] == value]
    #         temp_target_hc = df_n[df_n['specific.disorder'] == 6.0]
    #         temp_target = pd.concat([temp_target_ill, temp_target_hc], axis=0)
    #         temp_target.loc[temp_target['specific.disorder'] == 6.0, 'specific.disorder'] = 'Control'
    #         temp_target.loc[temp_target['specific.disorder'] == value, 'specific.disorder'] = 'Target'
    #         temp_target.to_csv('base_data\EEG.machinelearing_data_BRMH_sub_' + str(value) +  '.csv')
    # exit()
    ########################################################
    # # 数据集划分，仅使用训练集和测试集的情况
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, stratify=y)    
    # 特征选择
    # # 参数设置
    # opts = {'k': 5, 'fold': {'xt': X_train, 'yt': y_train, 'xv': X_test, 'yv': y_test}, 'N': 10, 'T': 100}
    # # 执行特征选择
    # fmdl = jfs(X, y, opts)
    # sf = fmdl['sf']  # 已选特征的索引
    # X_train = X_train[:, sf]
    # X_test = X_test[:, sf]
    # 样本均衡
    # adasyn = ADASYN(random_state=42)
    # X_train, y_train = adasyn.fit_resample(X_train, y_train)
    # 将y调整为独热编码
    ############################SVM性能评估################################
    # svm_model = SVC(kernel='linear')
    # svm_model.fit(X_train, y_train)
    # y_pred = svm_model.predict(X_test)
    # print(metrics.classification_report(y_test, y_pred))
    # with open('classification_report.txt', 'a') as file:
    #     file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "----SVM Results:")
    #     file.write(metrics.classification_report(y_test, y_pred))
    # ############################随机森林性能评估############################
    # rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
    # rf_classifier.fit(X_train, y_train)
    # y_pred = rf_classifier.predict(X_test)
    # print(metrics.classification_report(y_test, y_pred))
    # with open('classification_report.txt', 'a') as file:
    #     file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "----RandomForest Results:")
    #     file.write(metrics.classification_report(y_test, y_pred))
    # ############################XGBoost性能评估############################
    # model = get_xgboost()
    # # 训练模型
    # model.fit(X_train, y_train)
    # # 对测试集进行预测
    # y_pred = model.predict(X_test)
    # print(metrics.classification_report(y_test, y_pred))
    # with open('classification_report.txt', 'a') as file:
    #     file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "----XGBoost Results:")
    #     file.write(metrics.classification_report(y_test, y_pred))
    # ############################LightGBM性能评估###########################
    # # 不同分类等级需要调整params中的类别数
    # dtrain = lgb.Dataset(X_train, y_train)
    # gbm = lgb.train(get_lgb_param(), dtrain)
    # y_pred = gbm.predict(X_test).tolist()
    # y_pred = [1 if x > 0.5 else 0 for x in y_pred]
    # print(metrics.classification_report(y_test, y_pred))
    # with open('classification_report.txt', 'a') as file:
    #     file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "----LightGBM Results:")
    #     file.write(metrics.classification_report(y_test, y_pred))
    ############################DSSM性能评估###########################
    # exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_file', default=r'./model_params/')
    parser.add_argument('--gpu', default=True, help='use GPU')
    parser.add_argument('--info_file', default=r'./model_info/')
    parser.add_argument('--model_dir', default=r'./model_save/')
    # parser.add_argument('--model_dir', default=r'./experiment_04_12class_res/ALL-Model/')
    parser.add_argument('--restore_file', default=None)
    # 获取参数设置
    args = parser.parse_args()
    json_path = os.path.join(args.setting_file, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = tool.Params(json_path)
    params.model_dir = args.model_dir
    params.restore_file = args.restore_file
    
    # 测试GPU是否可用
    if torch.cuda.is_available():
        params.gpu = args.gpu
        print('gpu is available')
    else:
        params.gpu = False
        print('gpu is not available!')
        
    # 设置随机数种子用于复现实验结果
    torch.manual_seed(230)
    if params.gpu != False:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(230)

    # 设置打印输出消息
    tool.set_logger(os.path.join(args.info_file, 'train.log'))
    # tensorboard设置
    tb_writer = tool.TensorBoardWriter(args.info_file)
    # 精神障碍类别数目
    num_classes = len(np.unique(y))
    # 输出不同类别的标签
    metric_labels = np.unique(y)
    # 输入/输出数据的维度
    input_dim = X.shape[1]
    output_dim = len(np.unique(y))
    # output_dim = 1
    # 生成数据加载器
    # data = prepare_datasets(X_train, y_train, X_test, y_test, params.train_ratio, params.val_ratio, params.batch_size, scaler=scaler)
    # SHAP分析使用
    # 只分析对疾病影响较大的特征
    data = prepare_datasets(X_train, y_train, X_test, y_test, params.train_ratio, params.val_ratio, params.batch_size, scaler=scaler, X=X, y=y, Mode='Explain')
    # data_loader = data['loaders']['data']
    # data_sets = data['datasets']['data']
    #########################################
    # train_loader = data['loaders']['train']
    # test_loader = data['loaders']['test']
    # 针对验证集和测试集使用同一个的情况，这里也就是训练集:测试集=7:3
    # val_loader = test_loader
    # 构建模型进行实验
    dropout = 0.1
    model = DSSMClassifier(input_dim, output_dim, dropout)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.lr,
                                momentum=0.9,
                                weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.lr,
                                weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")
    # 动态调整学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6, verbose=True)
    # 重新训练神经网络并进行测试
    # train_and_evaluate(model,
    #                     train_data=train_loader,
    #                     val_data=val_loader,
    #                     optimizer=optimizer,
    #                     scheduler=scheduler,
    #                     params=params,
    #                     metric_labels=metric_labels,
    #                     model_dir=params.model_dir,
    #                     tb_writer=tb_writer,
    #                     restore_file=params.restore_file)
    # restore_path = os.path.join(params.model_dir, 'best.pth.tar')
    # tool.load_checkpoint(restore_path, model, optimizer=None)
    # evaluate(model, test_loader, metric_labels, mode='Test')
    #########################使用SHAP解释模型#############################
    # exit()
    restore_path = os.path.join(params.model_dir, model_name)
    tool.load_checkpoint(restore_path, model, optimizer=None)
    X = torch.from_numpy(X).float()
    explainer = shap.GradientExplainer(model, X)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[shap_index], X, cmap='rainbow', feature_names=df_x.columns, max_display=10, show=False)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('SHAP value (impact on model output)', fontsize=20)
    plt.show()
    feature_idx = df_x.columns.get_loc(columns_name)
    deg = 1 # 选择拟合的多项式阶数为3
    coefficients = np.polyfit(X[:, feature_idx], shap_values[shap_index][:, feature_idx], deg) # 进行多项式拟合
    poly_eq = np.poly1d(coefficients)
    plt.scatter(X[:, feature_idx], shap_values[shap_index][:, feature_idx], alpha=0.5)
    plt.plot(X[:, feature_idx], poly_eq(X[:, feature_idx]), color='orange', label='Fitted Curve')
    plt.axhline(0, color='gray', linewidth=0.8, ls='--')
    plt.xlabel(columns_name)
    plt.ylabel('SHAP Value for ' + columns_name)
    plt.show()