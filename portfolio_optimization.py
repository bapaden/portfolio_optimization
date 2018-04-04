import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import csv

# 0) Desired annual return. The script will minimize variance.
annual_return = 0.3

# 1) Load security historical prices and dividends
prices = np.genfromtxt ('prices.csv', delimiter=",")
dividends = np.genfromtxt ('dividend.csv', delimiter=",")
data_points = prices.shape[0]-1
num_stocks = prices.shape[1]
with open('prices.csv', 'rb') as csvfile:
  stocks = csv.reader(csvfile, delimiter=',', quotechar='|')
  stock_list = stocks.next()

print dividends

# 2) Calculate daily changes
delt = np.ndarray(shape=(data_points-1,num_stocks), dtype=float)
daily_div = np.exp(np.log(1+dividends[1,:])/365.0)-1.0
print daily_div
for stock_idx in range(0,num_stocks):
  for price_idx in range(1,data_points-1):
    delt[price_idx][stock_idx] = (prices[price_idx+1][stock_idx]-prices[price_idx][stock_idx])/prices[price_idx][stock_idx]


# 3) Compute sample mean and sample covariance for max-likelihood for normal
covariance = np.zeros(shape=(num_stocks,num_stocks),dtype=float)
mean = np.zeros(shape=(1,num_stocks),dtype=float)
for return_idx in range(1,data_points-1):
  mean = mean + delt[return_idx,:]
mean = mean/data_points
for return_idx in range(0,data_points-1):
  covariance = covariance + np.matmul(np.transpose(delt[return_idx,:]-mean),delt[return_idx,:]-mean)
covariance = covariance/data_points

print "Mean \n", mean
print "Covariance \n", covariance

# 4) Set up constraints for portfolio

#Equality constraints
Eq = np.ndarray(shape=(1,num_stocks),dtype=float)
eq = np.ndarray(shape=(1,1),dtype=float)
#Total investment is 100% of portfolio
Eq[0,:] = np.ones(shape=(1,num_stocks),dtype=float)
eq[0] = 1.0

#Inequality constraints
Ineq = np.ndarray(shape=(num_stocks+1,num_stocks),dtype=float)
ineq = np.ndarray(shape=(num_stocks+1,1),dtype=float)
for constr_idx in range(0,num_stocks):
  Ineq[constr_idx,:] = -np.identity(num_stocks)[constr_idx,:]
  ineq[constr_idx] = 0
print mean[0,:]
print np.exp(np.log(1+dividends[1,:])/365.0)-1.0
Ineq[num_stocks,:] = -(mean[0,:] + np.exp(np.log(1+dividends[1,:])/365.0)-1.0)
ineq[num_stocks] = -(np.exp(np.log(1.0+annual_return)/365.0)-1.0)

#min x'Qx+p'x
#s.t. Ax = b, Gx <= h 

#cost
Q = matrix(covariance)
p = matrix(np.zeros(shape=(num_stocks,1),dtype=float))
#Inequality constraints
G = matrix(Ineq)
h = matrix(ineq)
#Equality constraints
A = matrix(Eq)
b = matrix(eq)

#Solution from cvx
sol=solvers.qp(Q, p, G, h, A, b)

#Calculate variance of assets after one year
print "Allocation of assets \n", sol['x']

#Check that annual return is no less than desired value
r = (1.0+np.dot(mean,sol['x']))**365.0
print "Annual return \n", r
print "Variance of portfolio value \n", np.sqrt(365.0*sol['primal objective'])

# 5) Visualize solutions
plt.figure(1)
plt.xlabel('Day')
plt.ylabel('Relative value of security')
for stock_idx in range(0,num_stocks):
 plt.plot(prices[:,stock_idx]/prices[1][stock_idx],label=stock_list[stock_idx])
plt.legend()

plt.figure(2)
y_pos = np.arange(len(stock_list))
plt.bar(y_pos, sol['x'], align='center', alpha=0.5)
plt.xticks(y_pos, stock_list)
plt.ylabel('Optimal Portfolio Allocation')

plt.show()
