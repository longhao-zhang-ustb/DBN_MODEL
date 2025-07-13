from sklearn.datasets import load_breast_cancer
# from genetic_selection import GeneticSelectionCV
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.decomposition import PCA
# from sklearn.impute import KNNImputer
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, KMeansSMOTE, SVMSMOTE, SMOTE
from base_model.xgboost_base import get_xgboost
# import lightgbm as lgb
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
from data_preprocess.data_loader import prepare_datasets, prepare_oversampling_dataset
from model_pth.Fitness import train_and_evaluate
from model_pth.evaluate import evaluate
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks, NeighbourhoodCleaningRule, CondensedNearestNeighbour, OneSidedSelection
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import shap
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder,KBinsDiscretizer
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle

plt.rcParams['font.sans-serif'] = ['Times New Roman']

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
    ###第3、5类需要再测试
    df = pd.read_csv(r"base_data\EEG.machinelearing_data_BRMH_0.0.csv")
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
    df_y = df_y.map({'Target': 1, 'Control': 0}).astype(int)
    # X = df_x.values
    # data_visual(X, y)
    # exit()
    ###########将数据集划分为多个二分类数据############
    # # Healthy control: 2, Trauma and stress related disorder: 6, 
    # # Schizophrenia: 5, mood disorder: 3, Addictive disorder: 0
    # # Obessive compulsive disorder: 4,  Anxiety disorder: 1
    # df = pd.read_csv(r"base_data\EEG.machinelearing_data_BRMH.csv")
    # column = 'Unnamed: 122'
    # df = df.drop(columns=[column])
    # df_x = df.iloc[:, 8:]
    # df_y = df.iloc[:, 6]
    # y = df_y.values
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    # y = y.astype(np.float32)
    # df_n = pd.concat([df_x, df_y], axis=1)   
    # df_n['main.disorder'] = y 
    # unique_values, counts = np.unique(y, return_counts=True)
    # for value in unique_values:
    #     if value != 2.0:
    #         temp_target_ill = df_n[df_n['main.disorder'] == value]
    #         temp_target_hc = df_n[df_n['main.disorder'] == 2.0]
    #         temp_target_other = df_n.loc[(df_n['main.disorder'] != 2.0) & (df_n['main.disorder'] != value)]
    #         temp_target = pd.concat([temp_target_ill, temp_target_hc, temp_target_other], axis=0)
    #         temp_target.loc[temp_target['main.disorder'] == 2.0, 'main.disorder'] = 'Control'
    #         temp_target.loc[temp_target['main.disorder'] == value, 'main.disorder'] = 'Target'
    #         temp_target.loc[(temp_target['main.disorder'] != 'Control') & (temp_target['main.disorder'] != 'Target'), 'main.disorder'] = 'Other'
    #         temp_target.to_csv('base_data\EEG.machinelearing_data_BRMH_' + str(value) +  '_update3.csv')
    # exit()
    ########################################################
    # # 数据集划分，仅使用训练集和测试集的情况
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    ############################DSSM性能评估###########################
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting_file', default=r'./model_params/')
    parser.add_argument('--gpu', default=True, help='use GPU')
    parser.add_argument('--info_file', default=r'./model_info/')
    parser.add_argument('--model_dir', default=r'./model_save/')
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
    input_dim = df_x.shape[1]
    output_dim = len(np.unique(y))
    # 构建模型
    # 12类设置为0.1
    dropout = 0.1
    ############################机器学习性能评估################################
    svm_auc_scores = []
    svm_tpr_array = []
    svm_mean_fpr = np.linspace(0, 1, 100)  # 100个等间距FPR点
    rf_auc_scores = []
    rf_tpr_array = []
    rf_mean_fpr = np.linspace(0, 1, 100)  # 100个等间距FPR点
    xgb_auc_scores = []
    xgb_tpr_array = []
    xgb_mean_fpr = np.linspace(0, 1, 100)  # 100个等间距FPR点
    gbm_auc_scores = []
    gbm_tpr_array = []
    gbm_mean_fpr = np.linspace(0, 1, 100)  # 100个等间距FPR点
    our_auc_scores = []
    our_tpr_array = []
    our_mean_fpr = np.linspace(0, 1, 100)  # 100个等间距FPR点
    n = 0
    
    for train_index, test_index in kf.split(df_x):
        # 每一折创建一个新的模型
        svm_model = SVC(kernel='linear', probability=True)
        rf_classifier = RandomForestClassifier(n_estimators=300, random_state=42)
        xgboost_model = get_xgboost()
        lgb_model = LGBMClassifier()
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
            
        X_train_fold, X_test_fold = df_x.iloc[train_index], df_x.iloc[test_index]
        y_train_fold, y_test_fold = df_y.iloc[train_index], df_y.iloc[test_index]
        # 生成数据加载器
        data = prepare_datasets(np.array(X_train_fold), np.array(y_train_fold), np.array(X_test_fold), np.array(y_test_fold), params.train_ratio, params.val_ratio, params.batch_size, scaler=scaler)
        train_loader = data['loaders']['train']
        test_loader = data['loaders']['test']
        # 针对验证集和测试集使用同一个的情况，这里也就是训练集:测试集=7:3
        val_loader = test_loader
        
        # 在当前折上训练模型    
        svm_model.fit(X_train_fold, y_train_fold)
        rf_classifier.fit(X_train_fold, y_train_fold)
        xgboost_model.fit(X_train_fold, y_train_fold)
        gbm = lgb_model.fit(X_train_fold, y_train_fold)
        # Our Model
        train_and_evaluate(model,
                            train_data=train_loader,
                            val_data=val_loader,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            params=params,
                            metric_labels=metric_labels,
                            model_dir=params.model_dir,
                            tb_writer=tb_writer,
                            restore_file=params.restore_file)
        
        # 获取测试集的预测概率   
        svm_y_test_pred_prob = svm_model.predict_proba(X_test_fold)[:, 1]
        rf_y_test_pred_prob = rf_classifier.predict_proba(X_test_fold)[:, 1]
        xgboost_y_test_pred_prob = xgboost_model.predict_proba(X_test_fold)[:, 1]
        gbm_y_test_pred_prob = gbm.predict_proba(X_test_fold)[:, 1]
        restore_path = os.path.join(params.model_dir, 'best.pth.tar')
        tool.load_checkpoint(restore_path, model, optimizer=None)
        our_y_test_pred_prob, target_labels = evaluate(model, test_loader, metric_labels, mode='fold_5_test')
        our_y_test_pred_prob = our_y_test_pred_prob[:, 1]
        # 计算AUC分数    
        svm_fpr, svm_tpr, _ = roc_curve(y_test_fold, svm_y_test_pred_prob)
        rf_fpr, rf_tpr, _ = roc_curve(y_test_fold, rf_y_test_pred_prob)
        xgboost_fpr, xgboost_tpr, _ = roc_curve(y_test_fold, xgboost_y_test_pred_prob)
        gbm_fpr, gbm_tpr, _ = roc_curve(y_test_fold, gbm_y_test_pred_prob)
        our_fpr, our_tpr, _ = roc_curve(target_labels, our_y_test_pred_prob)
        
        svm_tpr_interp = np.interp(svm_mean_fpr, svm_fpr, svm_tpr)
        svm_tpr_interp[0] = 0
        svm_tpr_array.append(svm_tpr_interp)
        svm_auc_scores.append(auc(svm_fpr, svm_tpr))
        
        rf_tpr_interp = np.interp(rf_mean_fpr, rf_fpr, rf_tpr)
        rf_tpr_interp[0] = 0
        rf_tpr_array.append(rf_tpr_interp) 
        rf_auc_scores.append(auc(rf_fpr, rf_tpr))
        
        xgboost_tpr_interp = np.interp(xgb_mean_fpr, xgboost_fpr, xgboost_tpr)
        xgboost_tpr_interp[0] = 0
        xgb_tpr_array.append(xgboost_tpr_interp)
        xgb_auc_scores.append(auc(xgboost_fpr, xgboost_tpr))
        
        gbm_tpr_interp = np.interp(gbm_mean_fpr, gbm_fpr, gbm_tpr)
        gbm_tpr_interp[0] = 0
        gbm_tpr_array.append(gbm_tpr_interp) 
        gbm_auc_scores.append(auc(gbm_fpr, gbm_tpr))
        
        our_tpr_interp = np.interp(our_mean_fpr, our_fpr, our_tpr)
        our_tpr_interp[0] = 0
        our_tpr_array.append(our_tpr_interp)
        our_auc_scores.append(auc(our_fpr, our_tpr))
        n = n + 1
        print('完成第'+str(n)+'折交叉验证！')
    
    svm_tpr_mean = np.mean(svm_tpr_array, axis=0)
    svm_tpr_mean[-1] = 1.0 
    svm_auc_scores_mean = auc(svm_mean_fpr, svm_tpr_mean)
    svm_auc_scores_std = np.std(svm_auc_scores)
    
    rf_tpr_mean = np.mean(rf_tpr_array, axis=0)
    rf_tpr_mean[-1] = 1.0
    rf_auc_scores_mean = auc(rf_mean_fpr, rf_tpr_mean)
    rf_auc_scores_std = np.std(rf_auc_scores)
    
    xgboost_tpr_mean = np.mean(xgb_tpr_array, axis=0)
    xgboost_tpr_mean[-1] = 1.0
    xgboost_auc_scores_mean = auc(xgb_mean_fpr, xgboost_tpr_mean)
    xgboost_auc_scores_std = np.std(xgb_auc_scores)
    
    gbm_tpr_mean = np.mean(gbm_tpr_array, axis=0)
    gbm_tpr_mean[-1] = 1.0
    gbm_auc_scores_mean = auc(gbm_mean_fpr, gbm_tpr_mean)
    gbm_auc_scores_std = np.std(gbm_auc_scores)
    
    our_tpr_mean = np.mean(our_tpr_array, axis=0)
    our_tpr_mean[-1] = 1.0
    our_auc_scores_mean = auc(our_mean_fpr, our_tpr_mean)
    our_auc_scores_std = np.std(our_auc_scores)
    
    plt.figure()
    lw = 1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # 绘制SVM模型的ROC曲线
    plt.plot(svm_mean_fpr, svm_tpr_mean, lw=lw, label=f'SVM Mean ROC (AUC = {svm_auc_scores_mean:.2f} ± {svm_auc_scores_std:.2f})')
    plt.plot(rf_mean_fpr, rf_tpr_mean, lw=lw, label=f'Random Forest Mean ROC (AUC = {rf_auc_scores_mean:.2f} ± {rf_auc_scores_std:.2f})')
    plt.plot(xgb_mean_fpr, xgboost_tpr_mean, lw=lw, label=f'XGBoost Mean ROC (AUC = {xgboost_auc_scores_mean:.2f} ± {xgboost_auc_scores_std:.2f})')
    plt.plot(gbm_mean_fpr, gbm_tpr_mean, lw=lw, label=f'LightGBM Mean ROC (AUC = {gbm_auc_scores_mean:.2f} ± {gbm_auc_scores_std:.2f})')
    plt.plot(our_mean_fpr, our_tpr_mean, lw=2, label=f'Our Model Mean ROC (AUC = {our_auc_scores_mean:.2f} ± {our_auc_scores_std:.2f})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    plt.show()