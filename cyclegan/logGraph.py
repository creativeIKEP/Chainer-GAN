import matplotlib.pyplot as plt
import json
import numpy as np


def show_log_graph(log_path):
    file = open(log_path, "r")
    logJson = json.load(file)
    file.close()

    disALoss = []
    disBLoss = []
    genABLoss = []
    genBALoss = []

    for data in logJson:
        disALoss.append(float(data["d_A_loss"]))
        disBLoss.append(float(data["d_B_loss"]))
        genABLoss.append(float(data["g_AB_loss"]))
        genBALoss.append(float(data["g_BA_loss"]))

    x = np.array(range(len(logJson)))
    plt.plot(x, disALoss, label='d_A_loss')
    plt.plot(x, disBLoss, label='d_B_loss')
    plt.plot(x, genABLoss, label='g_AB_loss')
    plt.plot(x, genBALoss, label='g_BA_loss')
    plt.legend()
    plt.show()


def save_log_graph(log_path, save_path):
    file = open(log_path, "r")
    logJson = json.load(file)
    file.close()

    disALoss = []
    disBLoss = []
    genABLoss = []
    genBALoss = []

    for data in logJson:
        disALoss.append(float(data["d_A_loss"]))
        disBLoss.append(float(data["d_B_loss"]))
        genABLoss.append(float(data["g_AB_loss"]))
        genBALoss.append(float(data["g_BA_loss"]))

    x = np.array(range(len(logJson)))
    plt.plot(x, disALoss, label='d_A_loss')
    plt.plot(x, disBLoss, label='d_B_loss')
    plt.plot(x, genABLoss, label='g_AB_loss')
    plt.plot(x, genBALoss, label='g_BA_loss')
    plt.legend()
    plt.savefig(save_path)
