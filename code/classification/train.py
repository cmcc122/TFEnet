import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
current_dir = os.path.abspath(os.path.dirname(__file__))

classification_dir = os.path.join(current_dir, "..")  
sys.path.append(classification_dir)

from classification.model import tfenet
from classification.utils import datasets
from classification.utils.parse_config import read_json_config
from classification.utils.utils import *
from classification.utils.calculates import *
from classification.utils.logger import SingletonLogger  
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device, epoch, epochs):
    start = time.time()
    train_preds_total = []
    train_labels_total = []
    train_losses = 0.0
    train_acc = 0.0
    epoch_train_losses = []
    epoch_train_accs = []
    model.train()
    print("train")
    log = SingletonLogger.get_instance()
    for input, targets in dataloader:
        input, targets = input.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, targets)
        __, preds = outputs.max(1)
        train_preds_total.extend(preds.detach().cpu().numpy())
        train_labels_total.extend( targets.detach().cpu().numpy())

        loss.backward()
        
        #梯度裁剪
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0, norm_type=2)
        optimizer.step()
        acc = calculate_eeg_classification_accuracy(outputs.detach(), targets)
        train_losses += loss.item()
        train_acc += acc
        epoch_train_losses.append(loss.item())
        epoch_train_accs.append(acc)
    
    if lr_scheduler:
        lr_scheduler.step()

    avg_train_loss = train_losses / len(dataloader)
    avg_train_acc = train_acc / len(dataloader)
    print(f'Train Epoch [{epoch}/{epochs}], Avg Loss: {avg_train_loss:.4f}, Avg Acc: {avg_train_acc:.4f}')
    cr=classification_report(train_labels_total, train_preds_total,target_names=['0', '1'],digits=4,output_dict=True)
    cm=confusion_matrix(train_labels_total, train_preds_total)    

    print("Classification Report for Training Data:")
    print(cr)
    print("Confusion Matrix for Training Data:")
    print(cm) 

    dc = pd.DataFrame(cm).transpose()   
    dr = pd.DataFrame(cr).transpose()
    
    
    dr.to_csv(save_logs_folder + '/'+str(epoch) + '_train_classification_report.csv', index=True)
    dc.to_csv(save_logs_folder+  '/'+ str(epoch) + '_train_confusion_matrix.csv', index=True)

    
    finish = time.time()#
    print("训练的时间为{:.2f}".format(finish - start))   
    return avg_train_loss, avg_train_acc



def test_model(model, dataloader, criterion, device, epoch, epochs):

    start = time.time()
    losses = 0.0
    acc = 0.0
    epoch_test_losses = []
    epoch_test_accs = []
    test_preds_total = []
    test_labels_total = []
    model.eval()

    log = SingletonLogger.get_instance()


    for input, targets in dataloader:
        input, targets = input.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(input)
        loss = criterion(outputs, targets)
        current_acc = calculate_eeg_classification_accuracy(outputs.detach(), targets)
        losses += loss.item()
        acc += current_acc
        epoch_test_losses.append(loss.item())
        epoch_test_accs.append(current_acc)
        _, preds = outputs.max(1)
        test_preds_total.extend(preds.detach().cpu().numpy())
        test_labels_total.extend(targets.detach().cpu().numpy())
  
    avg_loss = losses / len(dataloader)
    avg_acc = acc / len(dataloader)
    print(f'Test Epoch [{epoch}/{epochs}], Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}')
    cr=classification_report(test_labels_total, test_preds_total,target_names=['0', '1',],digits=4,output_dict=True)
    cm=confusion_matrix(test_labels_total, test_preds_total) 

    print("Classification Report for Test Data:")
    print(cr)
    print("Confusion Matrix for Test Data:")
    print(cm)
    dr = pd.DataFrame(cr).transpose()
    dc = pd.DataFrame(cm).transpose()
    dr.to_csv(save_logs_folder + '/'+str(epoch) + '_test_classification_report.csv', index=True)
    dc.to_csv(save_logs_folder+ '/'+str(epoch) + '_test_confusion_matrix.csv', index=True)
    finish = time.time()
    print("评估的时间为{:.2f}".format(finish - start))

    return avg_loss, avg_acc
    
def train_and_evaluate_model(
        model, train_dataloader, test_dataloader,criterion, optimizer, lr_scheduler,
        epochs, device, checkpoint_enabled, save_weight_folder
):
   
    best_epoch=0
    best_test_acc = 0.0
    best_test_loss = float('inf')

    model.to(device)
    all_train_losses = []
    all_train_accs = []
    all_test_losses = []
    all_test_accs = []
    for epoch in range(1, epochs + 1):
        avg_train_loss, avg_train_acc  = train_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
           lr_scheduler=lr_scheduler,
            device=device,
            epoch=epoch, 
            epochs=epochs
        )
     
        avg_test_loss, avg_test_acc= test_model(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )
        
        all_train_losses.append(avg_train_loss)
        all_train_accs.append(avg_train_acc)
        all_test_losses.append(avg_test_loss)
        all_test_accs.append(avg_test_acc)

        if checkpoint_enabled:
            checkpoint_path = os.path.join(save_weight_folder, f'checkpoint_epoch{epoch}.pth')
            save_model_checkpoint(model, checkpoint_path)

        if avg_test_loss < best_test_loss :
            best_test_loss = avg_test_loss
            
        if best_test_acc<avg_test_acc :
            best_test_acc=avg_test_acc
            best_epoch=epoch
            best_model_path = os.path.join(save_weight_folder, f'best_model.pth')
            save_model_checkpoint(model, best_model_path)

        print("第{:d}epoch为测试集最佳准确率{:.7f}".format(best_epoch,best_test_acc))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), all_train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), all_test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_logs_folder, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), all_train_accs, label='Train Accuracy')
    plt.plot(range(1, epochs + 1), all_test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)  
    accuracy_plot_path = os.path.join(save_logs_folder, 'accuracy_curve.png')
    plt.savefig(accuracy_plot_path)
    plt.close()
    
        
def easy_train_model(train_dataloader, test_dataloader, class_weights,pretraining_enabled, pretraining_model_path,
                     epochs, learning_rate, lr_decay_step, lr_decay_factor, checkpoint_enabled,
                     save_weight_folder, save_logs_folder, device):
    SingletonLogger.get_instance(os.path.join(save_logs_folder,'train.log'))
   

    model=tfenet.TFEnet(inc=61, class_num=2,si=128)

    if pretraining_enabled:
        load_model_checkpoint(model, pretraining_model_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=16, eta_min=2e-7)

    train_and_evaluate_model(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=epochs,
        device=device,
        checkpoint_enabled=checkpoint_enabled,
        save_weight_folder=save_weight_folder
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)
    custom_dict = read_json_config("E:\data\code\config\custom.json")
    assert custom_dict, "文件读取异常"
    training_environment_setting = custom_dict.get("training_environment_setting", {})
    training_process_setting = custom_dict.get("training_process_setting", {})
    hyperparameters = custom_dict.get("hyperparameters", {})
    pretraining_setting = custom_dict.get("pretraining", {})


    root_directory = training_environment_setting.get("dataset")


    checkpoint_enabled = training_process_setting.get("checkpoint_enabled", False)
    save_weight_folder = training_process_setting.get("save_weight_folder", 'weights')
    save_logs_folder = training_process_setting.get("save_logs_folder", 'logs')
    trainset_radio = training_process_setting.get("trainset_radio", 0.8)

    pretraining_enabled = pretraining_setting.get("enabled", False)
    pretraining_model_path = pretraining_setting.get("model_path", 'weights/best_model.model')

    batch_size = hyperparameters.get("batch_size", 32)
    val_batch_size = hyperparameters.get("val_batch_size", 32)
    learning_rate = hyperparameters.get("lr", 0.001)
    epochs = hyperparameters.get("epochs", 10)
    
    lr_decay_step = hyperparameters.get("lr_decay_step", 1)
    lr_decay_factor = hyperparameters.get("lr_decay_factor", 0.99)

    os.makedirs(save_weight_folder, exist_ok=True)
    os.makedirs(save_logs_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader , class_weights= datasets.load_and_process_dataset(
        root_directory=root_directory,
        batch_size=batch_size,
        val_batch_size=val_batch_size, 

    )

    easy_train_model(
        train_dataloader=train_dataloader, 
        class_weights=class_weights,
        test_dataloader=test_dataloader,
        pretraining_enabled=pretraining_enabled,
        pretraining_model_path=pretraining_model_path,
        epochs=epochs,
        learning_rate=learning_rate,
        lr_decay_step=lr_decay_step,
        lr_decay_factor=lr_decay_factor,
        checkpoint_enabled=checkpoint_enabled,
        save_weight_folder=save_weight_folder,
        save_logs_folder=save_logs_folder,
        device=device
    )