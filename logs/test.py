import math
import matplotlib.pyplot as plt
import numpy as np
values = [3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

for i in values:
    fileName = ['./{}_originalBSR.txt'.format(i), './{}_predictedBSR.txt'.format(i), './{}_predictedFeedbackBSR.txt'.format(i)]

    originalBSR=[]
    predictedBSR=[]
    predictedFeedbackBSR=[]

    file = open(fileName[0], "r")
    lines = file.readlines()
    for line in lines:
        originalBSR.append(int(line))
    file.close()

    file = open(fileName[1], "r")
    lines = file.readlines()
    for line in lines:
        predictedBSR.append(int(line))
    file.close()

    file = open(fileName[2], "r")
    lines = file.readlines()
    for line in lines:
        predictedFeedbackBSR.append(int(line))
    file.close()

    plt.ylabel("BSR")
    plt.xlabel("Iterator")

    plt.figure()
    plt.plot(np.array(originalBSR), color='red')
    plt.savefig('{}_originalBSR.png'.format(i))

    plt.figure()
    plt.plot(np.array(predictedBSR), color='blue')
    plt.savefig('{}_predictedBSR.png'.format(i))

    plt.figure()
    plt.plot(np.array(predictedFeedbackBSR), color='green')
    plt.savefig('{}_predictedFeedbackBSR.png'.format(i))

    # print(originalBSR, predictedBSR, predictedFeedbackBSR)
    # Adding the title    
    plt.title("BSR Predictions")
                                        
    # Adding the labels

    # plt.show()

# for i in range(1,21):
#     fileName = ['./{}_originalBSR.txt', './{}_predictedBSR.txt', './{}_predictedFeedbackBSR.txt'.format(i+1, i+1, i+1)]

#     originalBSR=[]
#     predictedBSR=[]
#     predictedFeedbackBSR=[]

#     file = open(fileName[0], "r")
#     lines = file.readlines()
#     for line in lines:
#         originalBSR.append(int(line))
#     file.close()

#     file = open(fileName[1], "r")
#     lines = file.readlines()
#     for line in lines:
#         predictedBSR.append(int(line))
#     file.close()

#     file = open(fileName[2], "r")
#     lines = file.readlines()
#     for line in lines:
#         predictedFeedbackBSR.append(int(line))
#     file.close()

#     plt.ylabel("BSR")
#     plt.xlabel("Iterator")

#     plt.figure()
#     plt.plot(np.array(originalBSR), color='red')
#     plt.savefig('{}_originalBSR.png'.format(i+1))

#     plt.figure()
#     plt.plot(np.array(predictedBSR), color='blue')
#     plt.savefig('{}_predictedBSR.png'.format(i+1))

#     plt.figure()
#     plt.plot(np.array(predictedFeedbackBSR), color='green')
#     plt.savefig('{}_predictedFeedbackBSR.png'.format(i+1))

#     # print(originalBSR, predictedBSR, predictedFeedbackBSR)
#     # Adding the title    
#     plt.title("BSR Predictions")
                                        
#     # Adding the labels

#     # plt.show()