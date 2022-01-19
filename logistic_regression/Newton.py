import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

scores = []
is_admitted = []

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
        is_admitted.append(np.double(elt[0]))

is_admitted = np.mat(is_admitted).T
exam1_score = scores[:, 1]
exam2_score = scores[:, 2]
# 前40个admitted
is_admitted_score1 = exam1_score[0: 40, :]
is_admitted_score2 = exam2_score[0: 40, :]

not_admitted_score1 = exam1_score[40:, :]
not_admitted_score2 = exam2_score[40:, :]

plt.subplot(1, 2, 1)
plt.scatter(is_admitted_score1.T.tolist(), is_admitted_score2.T.tolist(), label = 'admitted', s = 20, marker='+')
plt.scatter(not_admitted_score1.T.tolist(), not_admitted_score2.T.tolist(), label = 'not admitted', s = 10, marker='o')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

theta = np.mat([0, 0, 0]).T
iterations = []
L_theta = []
A = np.zeros([80, 80])

for i in range(8) :
    # print(i)
    deviation = is_admitted - expit(scores * theta)
    H_theta = expit(scores * theta)
    J_theta = (scores.T * deviation)
    for j in range(80) :
        A[j, j] = H_theta[j, 0] * (1 - H_theta[j, 0])
    Hessian = scores.T * A * scores
    theta = theta + Hessian.I * J_theta
    print(theta)

    L_theta_i = - (is_admitted.T * (np.log(expit(scores * theta))) + \
    (np.ones([80, 1]) - is_admitted).T * (np.log(1 - expit(scores * theta)))) / 80
    iterations.append(i)
    L_theta.append(L_theta_i[0, 0])

print(theta)
print(L_theta[len(L_theta) - 1])
x = []
y = []
for i in range(15, 60):
    y.append(-theta[1,0] / theta[2, 0] * i - theta[0, 0] / theta[2, 0])
    x.append(i)

plt.legend()
plt.plot(x, y, color='g')

plt.subplot(1, 2, 2)
plt.plot(iterations, L_theta)
plt.title('Loss')
plt.xlabel('iterations')
plt.show()