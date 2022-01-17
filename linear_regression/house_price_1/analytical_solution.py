import numpy as np
import matplotlib.pyplot as plt
# x refers to year and a constant matrix
x = np.mat([[1.0, 2000], [1, 2001], [1, 2002], [1, 2003], [1, 2004], [1, 2005], [1, 2006], [1, 2007],
[1, 2008], [1, 2009], [1, 2010], [1, 2011], [1, 2012], [1, 2013]])

# y refers to house prices
y = np.mat([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]).T

theta = (np.linalg.inv(x.T * x)) * x.T * y

year2014 = np.mat([[1.0, 2014]])
estimated = year2014 * theta
print('theta=\n', theta)
print('estimated=\n', estimated[0, 0])

year = []

price = []
for i in range(len(x)) :
    price.append((x[i] * theta)[0, 0])
    year.append(x[i, 1])


plt.scatter(year, y.tolist())
plt.plot(year, price, c='#DC143C')
plt.xlim((1999,2014))
plt.title('Estimation')
plt.xlabel("year")
plt.ylabel("price")

plt.show()
