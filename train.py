# import stuff
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn import model_selection

import torch
import torch.utils.data as data
from itertools import product as product

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function

import librosa

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from torch.optim import lr_scheduler

from ghostnet_model import ghostnet,pretrained_ghostnet

import sklearn.metrics

import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,metavar='LR',help='initial learning rate')
parser.add_argument('--batch_size', default=8, type=int,metavar='N',help='mini-batch size (default: 8)')
parser.add_argument('-p','--pretrained', dest='pretrained', action='store_true',help='use pre-trained model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ghostnet',help='model architecture: (default: resnet18)')


logs = []
logs_eval = []
losses = []
accs = []
start = 0
epochs = 100

# watermark = "ghostnet"
# watermark = args.arch
# model_name = watermark


# load a wave data
def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x,fs

# change wave data to mel-stft
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()




#!数据增强
# data augmentation: add white noise
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


class ESC50Dataset(Dataset):
    def __init__(self, data, label, data_aug=False, _type='train'):
        self.label = label
        self.data_aug = data_aug
        self.data = data

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        label = self.label[idx]
        x, fs = load_wave_data(audio_dir, self.data[idx])

        # augumentations in wave domain.
        if self.data_aug:
            r = np.random.rand()
            if r < 0.3:
                x = add_white_noise(x)

            r = np.random.rand()
            if r < 0.3:
                x = shift_sound(x, rate=1+np.random.rand())

            r = np.random.rand()
            if r < 0.3:
                x = stretch_sound(x, rate=0.8+np.random.rand()*0.4)

        # convert to melsp
        melsp = calculate_melsp(x)

        # normalize
        mean = np.mean(melsp)
        std = np.std(melsp)

        melsp -= mean
        melsp /= std

        melsp = np.asarray([melsp, melsp, melsp])
        return melsp, label

# #!加载预训练模型
# # backbone
# import pretrainedmodels
# basemodel = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
# basemodel = nn.Sequential(*list(basemodel.children())[:-2])


# class mymodel(nn.Module):
#     def __init__(self):
#         super(mymodel, self).__init__()
#         self.features = basemodel
#         if model_name == "resnet34" or model_name == "resnet18":
#             num_ch = 512
#         else:
#             num_ch = 2048
#         self.fc = nn.Conv2d(num_ch, 50, 1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Sequential(
#             nn.Conv2d(4, 64, 3, stride=2,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, 3, stride=2,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, 3, stride=2,padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 64, 3, stride=2,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.rnn = nn.GRU(128, 128, 2, batch_first=True, bidirectional=True)
#         self.sed = nn.Sequential(
#             nn.Linear(64*256, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 50)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         # print(x.shape)
#         x = x.permute(0, 2, 1, 3)
#         # print(x.shape)
#         x = self.conv(x)
#         x = x.reshape(x.shape[0],x.shape[1], -1)
#         # print(x.shape)
#         x ,_= self.rnn(x)
#         # print(x.shape)
#         x = x.reshape(x.shape[0], -1)
#         x = self.sed(x)
#         return x


def train(epoch):
    model.train()
    print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_acc = 0.0
    acc = 0.0
    acc5 = 0.0

    t = tqdm(train_loader)

    for idx, (inputs,labels) in enumerate(t):
        # send to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # set opt
        optimizer.zero_grad()

        # run model
        outputs = model(inputs.float())


        loss = criterion(outputs,labels)
        # misc
        running_loss += loss
        running_acc += (outputs.argmax(1)==labels).float().mean()
        acc += (outputs.argmax(1)==labels).float().mean()
        #计算top5的准确率
        _,pred = outputs.topk(5,1,True,True)
        pred = pred.t()
        correct = pred.eq(labels.view(1,-1).expand_as(pred))
        correct_k = correct[:5].reshape(-1).float().sum(0)
        acc5 += correct_k

        loss.backward()
        optimizer.step()

        t.set_description(f'(loss={running_loss/(idx+1):.4f})(acc 1={acc/(idx+1):.4f})')
        if idx%8==7:
            rd = np.random.rand()

    #scheduler.step()
    losses.append(running_loss/len(train_loader))
    accs.append(running_acc/(len(train_loader)))
    print('train acc : {:.2f}%'.format(running_acc/(len(train_loader))*100))
    print('train loss : {:.4f}'.format(running_loss/len(train_loader)))
    print('dev acc1 : {:.2f}%'.format(acc/(len(train_loader))*100))
    print('dev acc5 : {:.2f}%'.format(acc5/(len(train_loader)*8)*100))

    # save logs
    log_epoch = {'epoch': epoch+1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                     'loss': running_loss/len(train_loader), "acc 1": acc/(len(train_loader)), "acc 5": acc5/(len(train_loader)*8)}
    logs.append(log_epoch)
    df = pd.DataFrame(logs)
    df.to_csv("log/log_output_train_{}.csv".format(watermark))

def eval(epoch):
    model.eval()
    print('epochs {}/{} '.format(epoch+1,epochs))
    running_loss = 0.0
    running_acc = 0.0
    acc = 0.0
    acc5 = 0.0
    t = tqdm(test_loader)

    for idx, (inputs,labels) in enumerate(t):
        # send to gpu
        inputs = inputs.to(device)
        labels = labels.to(device)

        # set opt
        optimizer.zero_grad()

        with torch.no_grad():
            # run model
            outputs = model(inputs.float())

        loss = criterion(outputs,labels)
        # misc
        running_loss += loss
        running_acc += (outputs.argmax(1)==labels).float().mean()
        acc += (outputs.argmax(1)==labels).float().mean()
        #计算top5的准确率
        _,pred = outputs.topk(5,1,True,True)
        pred = pred.t()
        correct = pred.eq(labels.view(1,-1).expand_as(pred))
        correct_k = correct[:5].reshape(-1).float().sum(0)
        acc5 += correct_k
        #loss.backward()
        #optimizer.step()

        t.set_description(f'(loss={running_loss/(idx+1):.4f})(acc 1={acc/(idx+1):.4f})')
        if idx%8==7:
            rd = np.random.rand()

    #scheduler.step()
    losses.append(running_loss/len(test_loader))
    accs.append(running_acc/(len(test_loader)))
    print('eval acc : {:.2f}%'.format(running_acc/(len(test_loader))*100))
    print('eval loss : {:.4f}'.format(running_loss/len(test_loader)))
    print('dev acc1 : {:.2f}%'.format(acc/(len(test_loader))*100))
    print('dev acc5 : {:.2f}%'.format(acc5/(len(test_loader)*8)*100))

    # save logs
    log_epoch = {'epoch': epoch+1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                     'loss': running_loss/len(test_loader), "acc 1": acc/(len(test_loader)), "acc 5": acc5/(len(test_loader)*8)}
    logs_eval.append(log_epoch)
    df = pd.DataFrame(logs_eval)
    df.to_csv("log/log_output_eval_{}.csv".format(watermark))



if __name__ == "__main__":
    os.makedirs("log", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    batch_size = 8
    CONTINUE = False

    # define directories
    base_dir = "./"
    esc_dir = os.path.join(base_dir, "ESC-50-master")
    meta_file = os.path.join(esc_dir, "meta/esc50.csv")
    audio_dir = os.path.join(esc_dir, "audio/")

    # load metadata
    meta_data = pd.read_csv(meta_file)

    # save train?
    SAVE_DATA = True

    # get training dataset and target dataset
    x = list(meta_data.loc[:,"filename"])
    y = list(meta_data.loc[:, "target"])

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)


    traindataset = ESC50Dataset(x_train, y_train, data_aug=True)
    testdataset = ESC50Dataset(x_test, y_test, data_aug=False)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,shuffle=True, num_workers=0)

    global args
    args = parser.parse_args()
    if args.arch == "ghostnet":
        model = ghostnet()
    elif args.arch == "pretrained_ghostnet":
        model = pretrained_ghostnet()
    else:
        raise ValueError("Invalid model name")


    watermark = args.arch
    model_name = watermark

    lr = args.lr


    # model = mymodel()
    # model = res18()
    # model = ghostnet()
    #使用预训练的ghostnet
    # model = pretrained_ghostnet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # epochs = 100 # original 50
    epochs = args.epochs

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, min_lr=1e-3*1e-4, factor=0.33)



    if CONTINUE:
        #start = START
        model.load_state_dict(torch.load("./models/cutonly_se_resnext50_32x4d_cutmix_236_29epochs_saved_weights.pth"))

    for epoch in range(start, epochs):
        train(epoch)
        eval(epoch)
        if epoch %10==0:
            torch.save(model.state_dict(), './models/{}_{}epochs_saved_weights.pth'.format(watermark, epoch))