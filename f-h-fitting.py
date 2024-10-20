import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
from scipy.signal import find_peaks_cwt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from statistics import mean, stdev

fig, ax = plt.subplots()

# Load the data from CSV files
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

# Define the trough positions (manually provided)
troughs2 = [11.3, 16.2, 21.2, 27.7]
peaks = []

for i in range(len(currents)):
    currents[i] = currents[i][(voltages[i]>=8) & (voltages[i]<=32)]
    voltages[i] = voltages[i][(voltages[i]>=8) & (voltages[i]<=32)]

    currents[i] = savgol_filter(currents[i], 20, 4)

    # Create a dataframe to hold the voltage and current for fitting/plotting
    df = pd.DataFrame({'Voltage': voltages[i], 'Current': currents[i]})

    # Polynomial + Gaussian model
    poly_model = lmfit.models.PolynomialModel(2)  # Polynomial model
    model = lmfit.models.PolynomialModel(2)       # Start with quadratic (2nd degree polynomial)

    # Add Gaussian peaks for each trough
    for k in range(len(troughs2)):
        model += lmfit.models.GaussianModel(prefix=f"p{k}_")

    # Create parameters for the model
    params = model.make_params()

    # Set initial values for Gaussian peak parameters
    for k, V_i in enumerate(troughs2):
        params[f"p{k}_center"].set(V_i)   # Peak center
        params[f"p{k}_sigma"].set(1)      # Peak width
        params[f"p{k}_amplitude"].set(0.05)  # Peak amplitude

    # Set initial values for the polynomial coefficients
    for k in range(3):  # For a 2nd degree polynomial (c0, c1, c2)
        params[f"c{k}"].set(0.05)

    # Perform the fitting on the data
    result = model.fit(df['Current'], params=params, x=df['Voltage'])
    
    # Plot the fitted model
    plt.plot(df['Voltage'], result.best_fit, label=labels[i], color=colors[i])

    p=result.params
    print(p)
    peaks_0 = [p['p0_center'].value, p['p1_center'].value, p['p2_center'].value, p['p3_center'].value]
    peaks.append(peaks_0)


# Plot the original data
#plt.scatter(df['Voltage'], df['Current'], label='Trial 2 Data', color = "red", s=3)

peaks_t = []
peaks_f2 = []

peaks_t = np.append(peaks_t, mean([peaks[1][0], peaks[2][0], peaks[3][0], peaks[4][0]]))
peaks_t = np.append(peaks_t, mean([peaks[1][1], peaks[2][1], peaks[3][1], peaks[4][1], peaks[0][1]]))
peaks_t = np.append(peaks_t, mean([peaks[1][2], peaks[2][2], peaks[3][2], peaks[4][2], peaks[0][2]]))
peaks_t = np.append(peaks_t, mean([peaks[1][3], peaks[2][3], peaks[3][3], peaks[4][3], peaks[0][3]]))

peaks_f2 = np.append(peaks_f2, stdev([peaks[1][0], peaks[2][0], peaks[3][0], peaks[4][0]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][1], peaks[2][1], peaks[3][1], peaks[4][1], peaks[0][1]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][2], peaks[2][2], peaks[3][2], peaks[4][2], peaks[0][2]]))
peaks_f2 = np.append(peaks_f2, stdev([peaks[1][3], peaks[2][3], peaks[3][3], peaks[4][3], peaks[0][3]]))

for i in range(len(peaks_t)):

    if i == 0:
        plt.axvline(x=peaks_t[i], color='black', linestyle='--', label = "Average Peak Voltage", linewidth=2)
        plt.axvspan(peaks_t[i]-peaks_f2[i], peaks_t[i]+peaks_f2[i], color='orange', label = "$\sigma$ of Peak Data", alpha=0.5)
    else:
        plt.axvline(x=peaks_t[i], color='black', linestyle='--',  linewidth=2)
        plt.axvspan(peaks_t[i]-peaks_f2[i], peaks_t[i]+peaks_f2[i], color='orange', alpha=0.5)

print(peaks_f2)
print(peaks_t)

ax.text(0.04, 0.2, "11.80±0.08", transform=ax.transAxes, fontsize=11, va='top')
ax.text(0.21, 0.25, "16.53±0.19", transform=ax.transAxes, fontsize=11, va='top')
ax.text(0.40, 0.4, "21.39±0.06", transform=ax.transAxes, fontsize=11, va='top')
ax.text(0.61, 0.60, "27.02±0.47", transform=ax.transAxes, fontsize=11, va='top')

# Labeling the plot
plt.xlabel('Accelerating Voltage (V)')
plt.ylabel('Anode Current (nA)')
plt.grid()
ax.text(0.05, 0.95, "$χ^2$=0.013/256", transform=ax.transAxes, fontsize=14, va='top')
plt.legend(loc="lower right", prop={'size': 11})
plt.show()
