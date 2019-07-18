import matplotlib.pyplot as plt
import json
import numpy as np


def show_log_graph(log_path):
    file = open(log_path, "r")
    logJson = json.load(file)
    file.close()

    disLoss = []
    genLoss = []

    for data in logJson:
        disLoss.append(float(data["d_loss"]))
        genLoss.append(float(data["g_loss"]))

    x = np.array(range(len(logJson)))
    plt.plot(x, disLoss, label='d_loss')  # label で凡例の設定
    plt.plot(x, genLoss, label='g_loss')  # label で凡例の設定
    plt.legend()
    plt.show()


def save_log_graph(log_path, save_path):
    file = open(log_path, "r")
    logJson = json.load(file)
    file.close()

    disLoss = []
    genLoss = []

    for data in logJson:
        disLoss.append(float(data["d_loss"]))
        genLoss.append(float(data["g_loss"]))

    x = np.array(range(len(logJson)))
    plt.plot(x, disLoss, label='d_loss')  # label で凡例の設定
    plt.plot(x, genLoss, label='g_loss')  # label で凡例の設定
    plt.legend()
    plt.savefig(save_path)
