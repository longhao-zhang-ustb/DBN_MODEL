from model_pth.DSSM_Model import DSSMClassifier
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
import utils.tool as tool
from tqdm import trange
import torch.nn as nn
from model_pth.evaluate import evaluate
import numpy as np

def train(model, train_data, optimizer, scheduler):
    model.train()
    loss_avg = tool.RunningAverage()
    t = trange(len(train_data))
    train_iter = iter(train_data)
    for i in t:
        X_batch, y_batch = next(train_iter)
        # X_batch = X_batch.cuda()
        # y_batch = y_batch.cuda()
        outputs = model(X_batch)
        loss = model.loss(outputs, y_batch.long())
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        
        # 进行梯度裁剪，防止梯度爆炸或消失
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        
        # performs updates using calculated gradients
        optimizer.step()
        # update the average loss
        loss_avg.update(loss.cpu().item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    
    # scheduler.step()
    scheduler.step(loss_avg())
   
    return loss_avg()

def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, metric_labels, model_dir, tb_writer, restore_file):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        tool.load_checkpoint(restore_path, model, optimizer=None)
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(1, params.max_epoch + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.max_epoch))
        # Train for one epoch on training set
        train_loss = train(model, train_data, optimizer, scheduler)
        # Evaluate for one epoch on training set and validation set
        train_metrics = evaluate(model, train_data, metric_labels)
        train_metrics['loss'] = train_loss
        train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics:" + train_metrics_str)
        val_metrics = evaluate(model, val_data, metric_labels)
        val_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics: " + val_metrics_str)
        tb_writer.add_scalars('loss',
                              {'train': train_metrics['loss'],
                               'val': val_metrics['loss'], },
                              epoch)
        tb_writer.close()
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        # Save weights ot the network
        tool.save_checkpoint({'epoch': epoch + 1,
                              'state_dict': model.state_dict(),
                              'optim_dict': optimizer.state_dict()},
                             is_best=improve_f1 > 0,
                             checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.max_epoch:
            logging.info("best val f1: {:05.2f}".format(best_val_f1))
            break
