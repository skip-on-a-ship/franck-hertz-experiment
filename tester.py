import numpy as np

data1 = [5.13, 0.06, 0.05]
data2 = [4.81, 0.05, 0.10]
answer = [0,0,0]

combined_uncertainty1 = np.sqrt((data1[1])**2 + (data1[2]**2))
combined_uncertainty2 = np.sqrt((data2[1])**2 + (data2[2]**2))

answer[0] = (data1[0]*(1/combined_uncertainty1**2)+data2[0]*(1/combined_uncertainty2**2))/((1/combined_uncertainty1**2)+(1/combined_uncertainty2**2))

answer[1] = np.sqrt(((data1[1]/2))**2 + ((data2[1]/2)**2))

answer[2] = np.sqrt((data1[2])**2 + (data2[2]**2))

print(np.sqrt(data1[0]-data2[0]))

print(answer)