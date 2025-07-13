# 将多个二分类模型进行集成，构建多分类精神障碍诊断模型，并评估模型的性能
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from model_pth.DSSM_Model import DSSMClassifier
from sklearn.model_selection import train_test_split
from data_preprocess.data_loader import prepare_datasets
import os
import argparse
import utils.tool as tool
import torch
from sklearn.metrics import classification_report

def evaluate(model, test_data):
     # 将模型设置为评估模式
     model.eval()
     loss_avg = tool.RunningAverage()
     output_labels = list()
     target_labels = list()
     with torch.no_grad():
        for i, batch_data in enumerate(test_data):
            predict_x, labels = batch_data
            outputs = model(predict_x)
            batch_output_labels = torch.max(outputs, dim=1)[1]
            output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
            target_labels.extend(labels.data.cpu().numpy().tolist())
     
     return output_labels, target_labels

if __name__ == "__main__":
     # 忽略 DeprecationWarning
     warnings.filterwarnings("ignore", category=DeprecationWarning)
     parser = argparse.ArgumentParser()
     parser.add_argument('--setting_file', default=r'./model_params/')
     parser.add_argument('--gpu', default=True, help='use GPU')
     parser.add_argument('--info_file', default=r'./model_info/')
     parser.add_argument('--model_dir', default=r'./experiment_02_res/')
     parser.add_argument('--restore_file', default=None)
     # 获取参数设置
     args = parser.parse_args()
     json_path = os.path.join(args.setting_file, 'params.json')
     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
     params = tool.Params(json_path)
     params.model_dir = args.model_dir
     params.restore_file = args.restore_file
     ############################EEG数据降维############################
     df = pd.read_csv(r"base_data\EEG.machinelearing_data_BRMH.csv")
     column = 'Unnamed: 122'
     df = df.drop(columns=[column])
     ############################数据预处理##############################
     # 6大类+1HC
     df_x = df.iloc[:, 8:]
     df_y = df.iloc[:, 6]
     y = df.values[:, 6]
     le = LabelEncoder()
     y = le.fit_transform(y)
     y = y.astype(np.float32)
     # 数据标准化 
     scaler = MinMaxScaler()
     df_x = pd.DataFrame(scaler.fit_transform(df_x), columns=df_x.columns)
     X = df_x.values
     # # 数据集划分，仅使用训练集和测试集的情况
     X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, stratify=y)
     data = prepare_datasets(X_train, y_train, X_test, y_test, params.train_ratio, params.val_ratio, params.batch_size, scaler=scaler)
     test_loader = data['loaders']['test']
     # 加载不同二分类模型的参数，完成模型的构建
     # 输入/输出数据的维度
     input_dim = X.shape[1]
     metric_labels = np.unique(y)
     output_dim = 2
     dropout=0.1
     # 构建模型实例
     ADModel = DSSMClassifier(input_dim, output_dim, dropout)
     AXDModel = DSSMClassifier(input_dim, output_dim, dropout)
     MDModel = DSSMClassifier(input_dim, output_dim, dropout)
     OCModel = DSSMClassifier(input_dim, output_dim, dropout)
     SCModel = DSSMClassifier(input_dim, output_dim, dropout)
     TSRModel = DSSMClassifier(input_dim, output_dim, dropout)
     # 加载模型参数
     restore_path = os.path.join(params.model_dir, '00_best.pth.tar')
     tool.load_checkpoint(restore_path, ADModel, optimizer=None)
     restore_path = os.path.join(params.model_dir, '01_best.pth.tar')
     tool.load_checkpoint(restore_path, AXDModel, optimizer=None)
     restore_path = os.path.join(params.model_dir, '03_best.pth.tar')
     tool.load_checkpoint(restore_path, MDModel, optimizer=None)
     restore_path = os.path.join(params.model_dir, '04_best.pth.tar')
     tool.load_checkpoint(restore_path, OCModel, optimizer=None)
     restore_path = os.path.join(params.model_dir, '05_best.pth.tar')
     tool.load_checkpoint(restore_path, SCModel, optimizer=None)
     restore_path = os.path.join(params.model_dir, '06_best.pth.tar')
     tool.load_checkpoint(restore_path, TSRModel, optimizer=None)
     # 获取预测结果
     ad_out, ad_tar = evaluate(ADModel, test_loader)
     axd_out, axd_tar = evaluate(AXDModel, test_loader)
     md_out, md_tar = evaluate(MDModel, test_loader)
     oc_out, oc_tar = evaluate(OCModel, test_loader)
     sc_out, sc_tar = evaluate(SCModel, test_loader)
     tsr_out, tsr_tar = evaluate(TSRModel, test_loader)
     target_label = tsr_tar
     # 根据模型的F1值调整预测结果
     del_out = []
     for index, item in enumerate(ad_tar):
          if sc_out[index] == 1:
               del_out.append(5)
          elif md_out[index] == 1:
               del_out.append(3)
          elif tsr_out[index] == 1:
               del_out.append(6)
          elif ad_tar[index] == 1:
               del_out.append(0)
          elif oc_tar[index] == 1:
               del_out.append(4)
          elif axd_tar[index] == 1:
               del_out.append(1)
          else:
               del_out.append(2)
     print(classification_report(del_out, target_label))