import utils.tool as tool
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import numpy as np

def evaluate(model, test_data, metric_labels, mode='Train', category=0):
    # 将模型设置为评估模式
    model.eval()
    loss_avg = tool.RunningAverage()
    output_labels = list()
    target_labels = list()
    output_array = list()
    with torch.no_grad():
        for i, batch_data in enumerate(test_data):
            predict_x, labels = batch_data
            # predict_x = predict_x.cuda()
            # labels = labels.cuda()
            outputs = model(predict_x)
            output_array.append(outputs.tolist())
            loss = model.loss(outputs, labels.long())
            loss_avg.update(loss.cpu().item())
            
            batch_output_labels = torch.max(outputs, dim=1)[1]
            output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
            target_labels.extend(labels.data.cpu().numpy().tolist())
            
        p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average="macro", zero_division=1)
        p_r_f1 = {
            'precision': p_r_f1_s[0]*100,
            'recall': p_r_f1_s[1]*100,
            'f1': p_r_f1_s[2]*100,
            'loss': loss_avg()
        }
        
        if mode == 'Test':
            report = classification_report(target_labels, output_labels)
            with open('classification_report.txt', 'a') as file:
                file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "----OurModel Results:")
                file.write(report)
            test_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in p_r_f1.items())
            logging.info("- TEST metrics: " + test_metrics_str)
            print(report)
            # # labels = ['Addictive disorder', 'Anxiety disorder', 'Healthy control', 'Mood disorder', 'Obsessive compulsive disorder',
            # #         'Schizophrenia', 'Trauma and stress related disorder']
            # labels = [['Healthy control', 'Addictive disorder'], 
            #           ['Healthy control', 'Anxiety disorder'], 
            #           ['Healthy control', 'Mood disorder'], 
            #           ['Healthy control', 'Obsessive compulsive disorder'], 
            #           ['Healthy control', 'Schizophrenia'], 
            #           ['Healthy control', 'Trauma and stress related disorder']]
            # labels = [['Healthy control', 'Acute stress disorder'], 
            #           ['Healthy control', 'Adjustment disorder'], 
            #           ['Healthy control', 'Alcohol use disorder'], 
            #           ['Healthy control', 'Behavioral addiction disorder'], 
            #           ['Healthy control', 'Bipolar disorder'], 
            #           ['Healthy control', 'Depressive disorder'],
            #           ['Healthy control', 'Obsessive compulsive disorder'], 
            #           ['Healthy control', 'Panic disorder'], 
            #           ['Healthy control', 'Posttraumatic stress disorder'],
            #           ['Healthy control', 'Schizophrenia'], 
            #           ['Healthy control', 'Social anxiety disorder']]
            # cm = confusion_matrix(target_labels, output_labels)
            # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels[category], yticklabels=labels[category])
            # plt.title('Confusion Matrix')
            # plt.xlabel('Predicted labels')
            # plt.ylabel('True labels')
            # plt.xticks(rotation=30)
            # plt.show()
        
        if mode == 'fold_5_test':
            # report = classification_report(target_labels, output_labels)
            # test_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in p_r_f1.items())
            # logging.info("- TEST metrics: " + test_metrics_str)
            # print(report)
            # return target_labels, output_labels
            output_array = np.array(output_array).reshape(-1, 2)
            return np.array(output_array), target_labels
        
        return p_r_f1 