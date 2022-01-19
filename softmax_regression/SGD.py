import numpy as np
import matplotlib.pyplot as plt
import random
scores = []
is_admitted = []
def htheta_x(theta, j, x):
    return (np.exp(x * theta[:, j]) / (np.sum(np.exp(x * theta))))[0, 0]
    
with open('./ex4Data/ex4x.dat') as f:

    for line in f:
        elt = line.split()
        score = [1]
        for i in range(len(elt)) :
            score.append(np.double(elt[i].strip()))
        scores.append(score)
        
scores = np.mat(scores)

with open('./ex4Data/ex4y.dat') as f:

    for line in f:
        elt = line.split()
        is_admitted.append(np.int(float(elt[0])) ^ 1)

is_admitted = np.mat(is_admitted).T



exam1_score = scores[:, 1]
exam2_score = scores[:, 2]

is_admitted_score1 = exam1_score[0: 40, :]
is_admitted_score2 = exam2_score[0: 40, :]

not_admitted_score1 = exam1_score[40:, :]
not_admitted_score2 = exam2_score[40:, :]

plt.subplot(1, 2, 1)
plt.scatter(is_admitted_score1.T.tolist(), is_admitted_score2.T.tolist(), label = 'admitted', s = 20, marker='+')
plt.scatter(not_admitted_score1.T.tolist(), not_admitted_score2.T.tolist(), label = 'not admitted', s = 10, marker='o')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')



theta = np.mat([[-16, 0], [0.2, 0], [0.2, 0]])
alpha = 0.00001
iterations = []
L_theta = []

for i in range(1000) :
    L_theta_i = 0
    grad_0 = np.mat([0.0, 0, 0]).T
    random_index = random.randint(0, 79)
    if is_admitted[random_index, 0] == 0:
        grad_0 += ((1.0 - htheta_x(theta, 0, scores[random_index, :])) * scores[random_index, :]).T 
    else :
        grad_0 += (( - htheta_x(theta, 0, scores[random_index, :])) * scores[random_index, :]).T 
    theta[:, 0] += alpha * grad_0

    for j in range(80) :
        if is_admitted[j, 0] == 0 :
            L_theta_i += - np.log(htheta_x(theta, 0, scores[j, :])) / 80
        else :
            L_theta_i += - np.log(htheta_x(theta, 1, scores[j, :])) / 80

    iterations.append(i)
    L_theta.append(L_theta_i)
     

print(theta)
x = []
y = []
for i in range(15, 65):
    y.append(- theta[1,0] / theta[2, 0] * i - theta[0, 0] / theta[2, 0])
    x.append(i)



plt.legend()
plt.plot(x, y, color='g')

plt.subplot(1, 2, 2)
plt.plot(iterations, L_theta)
plt.title('Loss')
plt.xlabel('iterations')
plt.show()