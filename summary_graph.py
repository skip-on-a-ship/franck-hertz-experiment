import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

numbers = np.array([1,2,3,4])
voltages1 = np.array([11.35, 16.34, 21.37, 26.92])
energy_uncertainties1 = np.array([0.19, 0.08, 0.07, 0.17])

#voltages1 = [11.80084456, 16.53503332, 21.38962746, 27.01601925] 
#energy_uncertainties1 = [0.08322914, 0.19213154, 0.06391397, 0.46852336]

voltages2 = np.array([13.37, 19.02, 24.67, 29.91])
energy_uncertainties2 = np.array([0.58, 0.37, 0.70, 0.27])

def linear_fit(x, m, b):
    return m * x + b

params, covariance = curve_fit(linear_fit, numbers, voltages1, sigma=energy_uncertainties1, absolute_sigma=True)
slope, intercept = params
uncertainties = np.sqrt(np.diag(covariance))

print(print("Uncertainties: ", uncertainties))

predicted_energies = linear_fit(numbers, slope, intercept)
chi_squared = np.sum((voltages1 - predicted_energies) ** 2)
print("Chi-squared:", chi_squared)

print("Slope (h) = " + str(slope))
print("Intercept (W) = " + str(intercept))
x_fit = np.linspace(min(numbers), max(numbers), 100)
y_fit = linear_fit(x_fit, slope, intercept)
"""
plt.plot(x_fit, y_fit, linestyle='dashed', color="red", label=f"Peak Best Fit")

params, covariance = curve_fit(linear_fit, numbers, voltages2, sigma=energy_uncertainties2, absolute_sigma=True)
slope, intercept = params
uncertainties = np.sqrt(np.diag(covariance))

print(print("Uncertainties: ", uncertainties))

predicted_energies = linear_fit(numbers, slope, intercept)
chi_squared = np.sum((voltages1 - predicted_energies) ** 2)
print("Chi-squared:", chi_squared)

print("Slope (h) = " + str(slope))
print("Intercept (W) = " + str(intercept))
x_fit = np.linspace(min(numbers), max(numbers), 100)
y_fit = linear_fit(x_fit, slope, intercept)

plt.errorbar(numbers, voltages2, yerr=energy_uncertainties2, fmt='o', color = "red", markersize=2, capsize=5, label = "Peak Data") 
"""
plt.errorbar(numbers, voltages1, yerr=energy_uncertainties1, fmt='o', color = "blue", markersize=2, capsize=5, label = "Observed Data") 
plt.plot(x_fit, y_fit, linestyle='dashed', color="blue", label=f"Best Fit")


plt.xlabel('Peak Number', fontsize=12)
plt.ylabel('Accelerating Voltages (V)', fontsize=12)
ax.text(0.05, 0.95, "y=(5.13±0.06)x+(6.06±0.18)\n$χ^2$=0.14/3", transform=ax.transAxes, fontsize=14, va='top')
plt.legend(loc="lower right", prop={'size': 14})

plt.show()