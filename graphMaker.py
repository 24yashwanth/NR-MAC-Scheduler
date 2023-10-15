fileName = './log_5UE_5Sec.out'
file = open(fileName, "r")
lines = file.readlines()
afterPred = []
beforePred = []

for line in range(len(lines)):
    beforeIndex, afterIndex = 0, 0
    if('BSR after pred for UE' in lines[line] and 'BSR before pred for UE' in lines[line-1]):
        afterIndex = lines[line].index('BSR after pred for UE')
        lines[line] = lines[line].replace(" reported", '')
        afterPred.append(int(lines[line][afterIndex + len('BSR after pred for UE') + 3 : ]))

        beforeIndex = lines[line-1].index('BSR before pred for UE')
        lines[line-1] = lines[line-1].replace(" reported", '')
        beforePred.append(int(lines[line-1][beforeIndex + len('BSR before pred for UE') + 3 : ]))

import math
import matplotlib.pyplot as plt
import numpy as np

MSE = np.square(np.subtract(beforePred,afterPred)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error: {}".format(RMSE))

plt.figure()
plt.plot(np.array(afterPred), color='red')
plt.plot(np.array(beforePred), color='blue')
   
# Adding the title    
plt.title("BSR Predictions")
                                    
# Adding the labels
plt.ylabel("BSR")
plt.xlabel("Iterator")
plt.show()


