import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

#Load security prices
prices = np.genfromtxt ('data.csv', delimiter=",")
data_points = prices.shape[0]
num_stocks = prices.shape[1]

#Have a look at prices relative to first day
plt.figure(1)
plt.xlabel('Day')
plt.ylabel('Relative value of security')
for stock_idx in range(0,num_stocks):
 plt.plot(prices[:,stock_idx]/prices[0][stock_idx])

#Calculate daily changes
delt = np.ndarray(shape=(data_points-1,num_stocks), dtype=float)
for stock_idx in range(0,num_stocks):
  for price_idx in range(0,data_points-1):
    #Arithmetic stats gaussian distribution
    #delt[price_idx][stock_idx] = (prices[price_idx+1][stock_idx]-prices[price_idx][stock_idx])/prices[price_idx][stock_idx]
    #Geometric stats for log gaussian r.v.
    delt[price_idx][stock_idx] = np.log(prices[price_idx+1][stock_idx]/prices[price_idx][stock_idx])

#Have a look at daily change
plt.figure(2)
plt.xlabel('Day')
plt.ylabel('Geometric Return')
for stock_idx in range(0,num_stocks):
  plt.plot(delt[:,stock_idx])

covariance = np.zeros(shape=(num_stocks,num_stocks),dtype=float)
mean = np.zeros(shape=(1,num_stocks),dtype=float)

#Compute sample mean and sample covariance for max-likelihood for normal
for return_idx in range(0,data_points-1):
  mean = mean + delt[return_idx,:]
mean = mean/data_points
for return_idx in range(0,data_points-1):
  covariance = covariance + np.matmul(np.transpose(delt-mean),delt-mean)

covariance = covariance/data_points

print "Mean \n", mean
print "Covariance \n", covariance

#Set up constraints for portfolio

#1)Equality constraints
Eq = np.ndarray(shape=(2,num_stocks),dtype=float)
eq = np.ndarray(shape=(2,1),dtype=float)
#Average return is desired value
Eq[0,:] = mean[0,:] #x
eq[0] = 0.002
#Total investment is 100% of portfolio
Eq[1,:] = np.ones(shape=(1,num_stocks),dtype=float)
eq[1] = 1.0

#2)Inequality constraints
minus_id = -np.identity(num_stocks)
zero = np.zeros(shape=(num_stocks,1),dtype=float)

#min x'Qx+p'x
#s.t. Ax = b, Gx <= h 

#1) cost
Q = matrix(covariance)
p = matrix(zero)
#2) Inequality constraints
G = matrix(minus_id)
h = matrix(zero)
#3) Equality constraints
A = matrix(Eq)
b = matrix(eq)

sol=solvers.qp(Q, p, G, h, A, b)

print "Allocation of assets \n", sol['x']
print "Variance of portfolio value \n", sol['primal objective']

#value = 1.0
#for data_idx in range(0,data_points-2):
  ##print sol['x']
  ##print delt[data_idx,:]
  #print matrix(sol['x'])*(matrix(delt[data_idx,:])).T
  ##print np.dot(sol['x'],delt[data_idx,:])
  ##value = value*(mean*np.transpose(delt[data_idx,:]))

#plt.show()