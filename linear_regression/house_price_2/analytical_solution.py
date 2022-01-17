import numpy as np
import matplotlib.pyplot as plt
features = []
house_price = []
with open('housing_data.txt') as f:
    for line in f:
        elt = line.split()
        item = [1]
        for i in range(len(elt) - 1) :
            item.append(np.double(elt[i].strip()))
        house_price.append(np.double(elt[len(elt)-1].strip()))
        features.append(item)


features = np.mat(features)
house_price = np.mat(house_price).T

training_features = features[0:400, 0:]
training_price = house_price[0:400, 0:]

testing_features = features[400:, 0:]
testing_price = house_price[400:, 0:]


theta = (np.linalg.inv(training_features.T * training_features)) * training_features.T * training_price
estimated_price = testing_features * theta


sequence = np.arange(1, 107)


deviation = training_features * theta - training_price
loss = (deviation.T * deviation) / (2 * 400)
print(estimated_price.T)


plt.scatter(sequence, estimated_price.T.tolist(), label = "estimated", s = 10)
plt.scatter(sequence, testing_price.T.tolist(), label = "testing", s = 10)
print(theta)
plt.legend()

plt.show()

