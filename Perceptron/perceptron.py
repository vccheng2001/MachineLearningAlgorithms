# inputs: x1, x2
# weights: w1, w2, b


import numpy as np 

def activation(x):
    return 0 if x < 0 else 1

def main():
    x = np.matrix([[0,0],[0,1],[1,0],[1,1]])
    y = np.asarray([0,1,1,1])
    lr = 1
    num_iters = 100
    (N,M) = x.shape # num samples, num features
    (w, b) = perceptron(x, y, lr, num_iters, N, M)
    error_rate = test(x, y, w, b, N)

def test(x, y, w, b, N):
    error_count = 0
    for i in range(N):
        print(f"\nTest input {i}: {x[i]}")
        out = forward(w, x[i], b)
        y_hat = activation(out)
        print(f"Pred output {i}: {y_hat}")
        print(f"Actual output {i}: {y[i]}")
        if int(y_hat) != int(y[i]):
            error_count += 1 
    error_rate = error_count / N
    print(f"Error count: {error_count}")
    print(f"Error rate: {error_rate}")
    return error_rate


def perceptron(x, y, lr, num_iters, N, M):
    w = np.zeros(M)
    w = np.expand_dims(w, axis=0) # w1, w2
    print('w :',  w)
    b = 0 
    for i in range(1,num_iters+1): 
        print(f"Iteration {i}")
        # reversed order if even iteration 
        lst = [0,1,2,3] if i % 2 == 1 else [3,2,1,0]
        for j in lst: 
            # print(f"row {j}")
            # print(f"x[{j}]: {x[j]}")
            # print(f"w: {w}, b: {b}")
            out = forward(w,x[j],b)
            y_hat  = activation(out) # pred
            error = (y[j] - y_hat)
            # print(f"actual {y[j]}- pred {y_hat} = error {error}")
            # print('stepppp', lr * x[j] * error)
            w = w + (lr * x[j] * error) 
            b = b + (lr * 1 * error)
            print(f"new w: {w}, new b: {b}")
            # print('\n')
    print("Finished training")
    print(f"w: {w}, b: {b}")
    return (w, b)
    
def forward(w,x,b):
    return np.dot(x, w.T).item() + b
    

if 'name == __main__':
    main()