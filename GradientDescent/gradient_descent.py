import numpy as np
def gradientDescent(x, X, y, theta, alpha, numIterations):
    for i in range(0, numIterations):
        # loss: X^T(theta)
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y #wxi + b - yi  [-1 -4 -3 -4 -5]
        N = len(y) # N = 5
        # calculate gradient wrt b, gradient wrt w
        gb = 2/N * np.sum(loss) #2/N * sum([-1 -4 -3 -4 -5])
        gw = 2/N * np.sum(x * loss) # 2/N * sum([-1 -8 -9 -16 -25])
        #print(gw)
        # update weights b, w
        theta = np.array([(theta[0]) -alpha * gb, (theta[1]) - alpha * gw]) 
        print('b:' + str(theta[0]) + " w:" + str(theta[1]))
        # calculate new cost 
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y #wxi + b - yi  [-1 -4 -3 -4 -5] 
        cost = (1/N) * np.sum(loss ** 2)
        print('cost...' + str(cost))
    return theta

numPoints = 5
X = np.zeros(shape=(numPoints, 2))
for i in range(0, numPoints):
        X[i][0] = 1
        X[i][1] = i+1
x = np.array([1,2,3,4,5])
y = np.array([3,8,9,12,15])
theta = np.array([0,2])
alpha = 0.01
numIterations = 100
gradientDescent(x,X,y,theta,alpha, numIterations)