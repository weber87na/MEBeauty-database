# train, test on MEBeauty and plot the results

import pytorch_mebeauty_dataset
import argparse
import copy
import os
from progiter import ProgIter
import torch
from torchvision import models
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
import pandas as pd



def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def run_epoch(model, loss_func, dataloader, device, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    total = 0

    for xb, yb in ProgIter(dataloader):
        xb = xb.to(device)
        yb = yb.to(device).float()

        with torch.set_grad_enabled(is_train):
            preds = model(xb).reshape(-1)
            loss = loss_func(preds, yb)

            if is_train:
                opt.zero_grad()
                loss.backward()
                opt.step()

        loss_sum += loss.item() * len(yb)
        total += len(yb)

    return loss_sum / total


def fit(model, loss_func, opt, train_dl, valid_dl, device = "cpu", epochs = 10,
        model_dir = "./pytorch_trained_models", patience = 5, scheduler = None):
    
    
    train_losses = []
    val_losses = []
    os.makedirs(model_dir, exist_ok=True)

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    
    epochs_without_improvement = 0
    history = []
    
    for epoch in range(epochs):
        
        print("epoch {}/{}".format(epoch + 1, epochs))
        train_loss = run_epoch(model, loss_func, train_dl, device, opt=opt)
        print('train loss {:.3f}'.format(train_loss))    
        train_losses.append(train_loss)

        val_loss = run_epoch(model, loss_func, valid_dl, device, opt=None)
        print('val loss {:.3f}'.format(val_loss))          
        val_losses.append(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": opt.param_groups[0]["lr"],
        })

        if scheduler is not None:
            scheduler.step(val_loss)
        
        if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(model_dir, 'best_model.pht'))
                print('Saved best model with val loss {:.3f}'.format(best_loss))
                epochs_without_improvement = 0
        else:
                print("Loss was better in a previous epoch; best val loss {:.3f}".format(best_loss))
                epochs_without_improvement += 1
        
        torch.save(model, os.path.join(model_dir, 'last_model.pht'))

        pd.DataFrame(history).to_csv(os.path.join(model_dir, "training_history.csv"), index=False)

        if epochs_without_improvement >= patience:
            print("Early stopping after {} epochs without improvement".format(patience))
            break
     
    model.load_state_dict(best_model_wts)
    return train_losses, val_losses


def set_trainable(module, trainable):
    for param in module.parameters():
        param.requires_grad = trainable


def regression_head(in_features, hidden_features=512, dropout=0.35):
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=hidden_features, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout, inplace=False),
        nn.Linear(in_features=hidden_features, out_features=1, bias=True),
    )


def model_preparation(base_model, device, lr = 0.0001, weight_decay = 0.0001,
                      dropout = 0.35, freeze_backbone = True, loss = "smoothl1"):
    
    if base_model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=False)
        in_features = model.fc.in_features
        if freeze_backbone:
            set_trainable(model, False)
        model.fc = regression_head(in_features, hidden_features=256, dropout=dropout)
        print("\n The model is fine-tuned on ResNet18 \n")

    elif base_model == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=False)
        in_features = model.fc.in_features
        if freeze_backbone:
            set_trainable(model, False)
        model.fc = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on ResNet50 \n")

    elif base_model == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT, progress=False)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            set_trainable(model.features, False)
        model.classifier = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on EfficientNet-B0 \n")

    elif base_model == 'densenet':   
        model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT, progress=False)
        in_features  = 2208 
        if freeze_backbone:
            set_trainable(model.features, False)
        model.classifier = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on DenseNet \n")
            
    elif base_model == 'mobilenet':   
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT, progress=False)
        in_features  = 1280 
        if freeze_backbone:
            set_trainable(model.features, False)
        model.classifier = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on MobileNet \n")
        
    elif base_model == 'alexnet':    
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT, progress=False)
        in_features  = 9216
        if freeze_backbone:
            set_trainable(model.features, False)
        model.classifier = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on AlexNet \n")
        
    else:  
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=False)
        in_features  = 25088
        if freeze_backbone:
            set_trainable(model.features, False)
        model.classifier = regression_head(in_features, hidden_features=512, dropout=dropout)
        print("\n The model is fine-tuned on VGG16 \n")

    model = model.to(device)
    if loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    return model, criterion, optimizer, scheduler

if __name__ == '__main__':
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_model', type=str, help='base model: resnet18, resnet50, efficientnet, densenet, mobilenet, alexnet, vgg16',
                         default = "resnet18")
    parser.add_argument('--train_augmentation', type=str_to_bool, help='train augmentation?',
                         default = False)
    parser.add_argument('--train_scores', type=str, help='csv file with scores for training',
                         default = 'scores/train_crop.csv')
    parser.add_argument('--test_scores', type=str, help='csv file with scores for validation',
                         default = 'scores/test_crop.csv')
    parser.add_argument('--batch_size', type=int, help='batch size',
                         default = 16)
    parser.add_argument('--epochs', type=int, help='number of epochs',
                         default = 50)
    parser.add_argument('--num_workers', type=int, help='number of workers',
                         default = 8)
    parser.add_argument('--pin_memory', type=int, help='pin_memory',
                         default = True)
    parser.add_argument('--lr', type=float, help='learning rate',
                         default = 0.0001)
    parser.add_argument('--weight_decay', type=float, help='Adam weight decay',
                         default = 0.0001)
    parser.add_argument('--dropout', type=float, help='dropout in regression head',
                         default = 0.35)
    parser.add_argument('--patience', type=int, help='early stopping patience',
                         default = 6)
    parser.add_argument('--loss', type=str, choices=["smoothl1", "mse"], help='training loss',
                         default = "smoothl1")
    parser.add_argument('--freeze_backbone', type=str_to_bool, help='freeze pretrained backbone?',
                         default = True)
    args = parser.parse_args()
    
    base_model = args.base_model
    train_aug = args.train_augmentation
    train_scores = args.train_scores
    test_scores = args.test_scores
    batch = args.batch_size
    epochs = args.epochs
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    patience = args.patience
    loss = args.loss
    freeze_backbone = args.freeze_backbone
   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cpu")
    print('The network is training on',device)
    traindata, testdata = pytorch_mebeauty_dataset.train_test_data(train_scores, test_scores, train_augmentation = train_aug, 
                                                           batch = batch, num_workers = num_workers, pin_memory = pin_memory) # train and test dataloaders
    
    model, criterion, optimizer, scheduler = model_preparation(
        base_model,
        device,
        lr = lr,
        weight_decay = weight_decay,
        dropout = dropout,
        freeze_backbone = freeze_backbone,
        loss = loss,
    )
    
    train_loss, val_loss = fit(model, criterion, optimizer, traindata, testdata, device, epochs, patience=patience, scheduler=scheduler)
    
    #mebeauty_dataset.plot_training(train_loss, val_loss)
    
    
