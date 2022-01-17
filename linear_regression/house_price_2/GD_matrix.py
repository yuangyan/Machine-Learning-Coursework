import numpy as np
import matplotlib.pyplot as plt
import time
features = []
house_price = []
with open('housing_data.txt') as f:
    for line in f:
        elt = line.split()
        item = [1]
        for i in range(len(elt) - 1) :
            item.append(np.double(elt[i].strip()))
        house_price.append(np.double(elt[len(elt) - 1].strip()))
        features.append(item)


features = np.mat(features)
house_price = np.mat(house_price).T

training_features = features[0:400, 0:]
training_price = house_price[0:400, 0:]

testing_features = features[400:, 0:]
testing_price = house_price[400:, 0:]



alpha = 0.0000005
# 梯度下降
# theta = np.mat([np.zeros(14)]).T
theta = np.mat([[2.8, -0.2, 0, 0, 1.71631351,  -1.5, 4.8, 0, -1.2, -0.4, 0, -0.8, 0, -0.5]]).T

iterations = []
losses = []
time_start = time.time()

for i in range(50) :
    deviation = training_features * theta - training_price
    grad = (training_features.T * deviation) / 400
    loss = (deviation.T * deviation) / (2 * 400)
    theta = theta - alpha * grad
    losses.append(loss[0, 0])
    iterations.append(i)

time_end = time.time() 

time_c= time_end - time_start   #运行所花时间
print('time cost', time_c, 's')

plt.subplot(1, 2, 1)
plt.plot(iterations, losses, label = "alpha = 5e-7")




alpha = 0.000005
# 梯度下降
# theta = np.mat([np.zeros(14)]).T
theta = np.mat([[2.8, -0.2, 0, 0, 1.71631351,  -1.5, 4.8, 0, -1.2, -0.4, 0, -0.8, 0, -0.5]]).T

iterations = []
losses = []

for i in range(50) :
    deviation = training_features * theta - training_price
    grad = (training_features.T * deviation) / 400
    loss = (deviation.T * deviation) / (2 * 400)
    theta = theta - alpha * grad
    losses.append(loss[0, 0])
    iterations.append(i)
plt.subplot(1, 2, 1)
plt.plot(iterations, losses, label = "alpha =5e-6")
plt.xlabel("iterations")
plt.ylabel("loss")
estimated_price = testing_features * theta



alpha = 0.0000001
# 梯度下降
# theta = np.mat([np.zeros(14)]).T
theta = np.mat([[2.8, -0.2, 0, 0, 1.71631351,  -1.5, 4.8, 0, -1.2, -0.4, 0, -0.8, 0, -0.5]]).T

iterations = []
losses = []

for i in range(50) :
    deviation = training_features * theta - training_price
    grad = (training_features.T * deviation) / 400
    loss = (deviation.T * deviation) / (2 * 400)
    theta = theta - alpha * grad
    losses.append(loss[0, 0])
    iterations.append(i)


plt.subplot(1, 2, 1)
plt.plot(iterations, losses, label = "alpha =1e-7")
plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend()


sequence = np.arange(1, 107)
plt.subplot(1, 2, 2)
plt.scatter(sequence, estimated_price.T.tolist(), label = "estimated", s = 10)
plt.scatter(sequence, testing_price.T.tolist(), label = "testing", s = 10)
plt.legend()

plt.show()
# print(estimated_price.T)









