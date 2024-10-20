import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from statistics import mean
from statistics import stdev 
from sklearn.metrics import mean_squared_error

current1 = np.array(pd.read_csv("trial1/current.csv")["  -0.61600"])
voltage1 = np.array(pd.read_csv("trial1/voltage1.csv")["  -0.96000"] - pd.read_csv("trial1/voltage2.csv")[" -51.20000"])

current2 = np.array(pd.read_csv("trial2/current.csv")["  -0.61600"])
voltage2 = np.array(pd.read_csv("trial2/voltage1.csv")["  -0.96000"] - pd.read_csv("trial2/voltage2.csv")[" -51.20000"])

current3 = np.array(pd.read_csv("trial3/current.csv")["  -0.61600"])
voltage3 = np.array(pd.read_csv("trial3/voltage1.csv")["  -0.96000"] - pd.read_csv("trial3/voltage2.csv")[" -51.20000"])

current4 = np.array(pd.read_csv("trial4/current.csv")["  -0.61600"])
voltage4 = np.array(pd.read_csv("trial4/voltage1.csv")["  -0.96000"] - pd.read_csv("trial4/voltage2.csv")[" -51.20000"])

current5 = np.array(pd.read_csv("trial5/current.csv")["  -0.61600"])
voltage5 = np.array(pd.read_csv("trial5/voltage1.csv")["  -0.96000"] - pd.read_csv("trial5/voltage2.csv")[" -51.20000"])

currents = [current1, current2, current3, current4, current5]
voltages = [voltage1, voltage2, voltage3, voltage4, voltage5]
peaks = []
troughs = []
colors = ["red","orange","green","blue","purple"]
labels = ["Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5"]
current1 = []

for i in range(len(currents)):
    currents[i] = currents[i][(voltages[i]>=8) & (voltages[i]<=32)]
    voltages[i] = voltages[i][(voltages[i]>=8) & (voltages[i]<=32)]

    current1 = []
    current1 = currents[i]
    currents[i] = savgol_filter(currents[i], 20, 4)

    sigma = 5
    currents[i] = gaussian_filter1d(currents[i], sigma)
    voltages[i] = gaussian_filter1d(voltages[i], sigma) 

    mse = mean_squared_error(current1, currents[i])

    peak = find_peaks_cwt(currents[i], widths=np.ones(currents[i].shape)*2)-1
    trough = find_peaks_cwt(-currents[i], widths=np.ones(currents[i].shape)*2)-1

    peak = np.delete(peak, [-1])
    if i==0: peak = np.delete(peak,[0])
    else: trough = np.delete(trough,[0])
    if not i==2: peak = np.delete(peak,[-1])

    if i==0: trough = np.delete(trough,[1])
    if i==1: trough = np.delete(trough,[0,2])
    if i==2: trough = np.delete(trough,[1,4])
    if i==4: trough = np.delete(trough,[0,2,4])

    peaks += [voltages[i][peak]]
    troughs += [voltages[i][trough]]

    plt.scatter(voltages[i], currents[i], s=2, color=colors[i], label = labels[i])

peaks_f = []
troughs_f = []
peaks_f2 = []
troughs_f2 = []

peaks_f = np.append(peaks_f, mean([peaks[1][0], peaks[2][0], peaks[3][0], peaks[4][0]]))
peaks_f = np.append(peaks_f, mean([peaks[1][1], peaks[2][1], peaks[3][1], peaks[4][1], peaks[0][0]]))
peaks_f = np.append(peaks_f, mean([peaks[1][2], peaks[2][2], peaks[3][2], peaks[4][2], peaks[0][1]]))
peaks_f = np.append(peaks_f, mean([peaks[1][3], peaks[2][3], peaks[3][3], peaks[4][3]]))

peaks_f2 = np.append(peaks_f2, stdev([peaks[1][0], peaks[2][0], peaks[3][0], peaks[4][0]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][1], peaks[2][1], peaks[3][1], peaks[4][1], peaks[0][0]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][2], peaks[2][2], peaks[3][2], peaks[4][2], peaks[0][1]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][3], peaks[2][3], peaks[3][3], peaks[4][3]]))

troughs_f = np.append(troughs_f, mean([troughs[1][0], troughs[2][0], troughs[3][0], troughs[4][0]]))
troughs_f = np.append(troughs_f, mean([troughs[1][1], troughs[2][1], troughs[3][1], troughs[4][1], troughs[0][0]]))
troughs_f = np.append(troughs_f, mean([troughs[1][2], troughs[2][2], troughs[3][2], troughs[4][2], troughs[0][1]]))
troughs_f = np.append(troughs_f, mean([troughs[1][3], troughs[2][3], troughs[3][3], troughs[4][3], troughs[0][2]]))

troughs_f2 = np.append(troughs_f2, stdev([troughs[1][0], troughs[2][0], troughs[3][0], troughs[4][0]]))
troughs_f2 = np.append(troughs_f2, stdev([troughs[1][1], troughs[2][1], troughs[3][1], troughs[4][1], troughs[0][0]]))
troughs_f2 = np.append(troughs_f2, stdev([troughs[1][2], troughs[2][2], troughs[3][2], troughs[4][2], troughs[0][1]]))
troughs_f2 = np.append(troughs_f2, stdev([troughs[1][3], troughs[2][3], troughs[3][3], troughs[4][3], troughs[0][2]]))

print(peaks_f)
print(peaks_f2)
print(troughs_f)
print(troughs_f2)

for i in range(len(peaks_f)):

    if i == 0:
        plt.axvline(x=peaks_f[i], color='black', linestyle='--', label = "Average Peak Voltage", linewidth=2)
        plt.axvspan(peaks_f[i]-peaks_f2[i], peaks_f[i]+peaks_f2[i], color='orange', label = "$\sigma$ of Peak/Trough Data", alpha=0.5)
    else:
        plt.axvline(x=peaks_f[i], color='black', linestyle='--',  linewidth=2)
        plt.axvspan(peaks_f[i]-peaks_f2[i], peaks_f[i]+peaks_f2[i], color='orange', alpha=0.5)



plt.xlabel('Accelerating Voltage (V)',fontsize=15)
plt.ylabel('Anode Current (nA)', fontsize=15)
plt.legend(prop={'size': 13})
plt.show()

#current_f = np.array(pd.read_csv("TEK0000_f.csv")["  -0.61600"])
#time_f = np.array(pd.read_csv("TEK0000_f.csv")["   0.000000000000"])

#plt.scatter(time_f, current_f, s=3)