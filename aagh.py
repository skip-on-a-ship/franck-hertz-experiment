import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

numbers = np.array([1,2,3])
voltages1 = np.array([4.99, 5.03, 5.55])
energy_uncertainties1 = np.array([0.20, 0.10, 0.20])

voltages2 = np.array([4.7, 4.9,5.6])
energy_uncertainties2 = np.array([0.2, 0.2, 0.5])

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
x_fit = np.linspace(0, max(numbers), 100)
y_fit = linear_fit(x_fit, slope, intercept)

plt.plot(x_fit, y_fit, linestyle='dashed', color="blue")

params, covariance = curve_fit(linear_fit, numbers, voltages2, sigma=energy_uncertainties2, absolute_sigma=True)
slope, intercept = params
uncertainties = np.sqrt(np.diag(covariance))

print(print("Uncertainties: ", uncertainties))

predicted_energies = linear_fit(numbers, slope, intercept)
chi_squared = np.sum((voltages1 - predicted_energies) ** 2)
print("Chi-squared:", chi_squared)

print("Slope (h) = " + str(slope))
print("Intercept (W) = " + str(intercept))
x_fit = np.linspace(0, max(numbers), 100)
y_fit = linear_fit(x_fit, slope, intercept)

plt.plot(x_fit, y_fit, linestyle='dashed', color="red")


plt.errorbar(numbers, voltages1, yerr=energy_uncertainties1, fmt='o', color = "blue", markersize=2, capsize=5, label = "CWT Method") 
plt.errorbar(numbers, voltages2, yerr=energy_uncertainties2, fmt='o', color = "red", markersize=2, capsize=5, label = "Multi-Gaussian Polynomial Method") 
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2)


plt.xlabel('Spacing Number', fontsize=12)
plt.ylabel('Excitation Potential $\Delta E$', fontsize=12)
#ax.text(0.05, 0.95, "y=(4.81±0.05)x+(6.96±0.13)\n$χ^2$=0.64/3", transform=ax.transAxes, fontsize=14, va='top')
plt.legend(loc="lower right", prop={'size': 14})

plt.show()