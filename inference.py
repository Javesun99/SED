import torch
from train import mymodel,load_wave_data,calculate_melsp,res18
import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import pandas as pd
import os

# display wave in heatmap
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()



def infer(path,file_name):
    model = res18()
    model.load_state_dict(torch.load('./models/resnet34_40epochs_saved_weights.pth'))
    model.eval()
    x,fs = load_wave_data(path,file_name)
    melsp = calculate_melsp(x)
    # normalize
    mean = np.mean(melsp)
    std = np.std(melsp)

    melsp -= mean
    melsp /= std

    melsp = np.asarray([melsp, melsp, melsp])
    melsp = melsp.reshape(1,melsp.shape[0],melsp.shape[1],melsp.shape[2])
    melsp = torch.from_numpy(melsp).float()
    with torch.no_grad():
        output = model(melsp)
        _,pred = torch.max(output,1)
    print(class_dict[pred.item()])

def print_audio_shape(file_path):
    #修改音频长度
    x, fs = librosa.load(file_path, sr=44100,duration=5.0)
    melsp = calculate_melsp(x)
    # show_melsp(melsp,fs)
    plt.figure(figsize=(5, 5))
    plt.imshow(melsp,aspect=10)
    plt.show()
    melsp = np.asarray([melsp, melsp, melsp])
    melsp = melsp.reshape(1,melsp.shape[0],melsp.shape[1],melsp.shape[2])
    melsp = torch.from_numpy(melsp).float()
    print(x.shape)



def infer2(file_path):
    model = res18()
    model.load_state_dict(torch.load('./models/resnet18_40epochs_saved_weights.pth'))
    model.eval()
    #修改音频长度
    x, fs = librosa.load(file_path, sr=44100,duration=5.0)
    melsp = calculate_melsp(x)
    # show_melsp(melsp,fs)
    # plt.figure(figsize=(5, 5))
    # plt.imshow(melsp,aspect=10)
    # plt.show()
    melsp = np.asarray([melsp, melsp, melsp])
    melsp = melsp.reshape(1,melsp.shape[0],melsp.shape[1],melsp.shape[2])
    melsp = torch.from_numpy(melsp).float()
    with torch.no_grad():
        output = model(melsp)
        _,pred = torch.max(output,1)
    print(pred.item())
    print(class_dict[pred.item()])


def infer3(file_path):
    model = res18()
    model.load_state_dict(torch.load('./models/resnet18_40epochs_saved_weights.pth'))
    model.eval()
    #修改音频长度
    x, fs = librosa.load(file_path, sr=44100,duration=5.0)
    melsp = calculate_melsp(x)#(128,1723)
    # normalize
    mean = np.mean(melsp)
    std = np.std(melsp)

    melsp -= mean
    melsp /= std

    melsp = np.asarray([melsp, melsp, melsp])
    melsp = melsp.reshape(1,melsp.shape[0],melsp.shape[1],melsp.shape[2])
    melsp = torch.from_numpy(melsp).float()
    with torch.no_grad():
        output = model(melsp)
        _,pred = torch.max(output,1)
    print(pred.item())
    print(class_dict[pred.item()])


if __name__ == '__main__':
    # define directories
    base_dir = "./"
    esc_dir = os.path.join(base_dir, "ESC-50-master")
    meta_file = os.path.join(esc_dir, "meta/esc50.csv")
    audio_dir = os.path.join(esc_dir, "audio/")
    # load metadata
    meta_data = pd.read_csv(meta_file)

    # get data size
    data_size = meta_data.shape
    # print(data_size)

    # arrange target label and its name
    class_dict = {}
    for i in range(data_size[0]):
        if meta_data.loc[i,"target"] not in class_dict.keys():
            class_dict[meta_data.loc[i,"target"]] = meta_data.loc[i,"category"]
    # load metadata
    meta_data = pd.read_csv(meta_file)
    # infer('./ESC-50-master/audio/','5-210540-A-13.wav')
    # infer('./','1.wav')
    # print_audio_shape('./ESC-50-master/audio/5-210540-A-13.wav')
    # print_audio_shape('./946684912.wav')
    infer3('./946684912.wav')
    # infer2('./ESC-50-master/audio/5-223176-A-37.wav')