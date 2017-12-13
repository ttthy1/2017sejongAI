def cost(W, X, y):
    s = 0
    for i in range(len(X)):
        s += (W * X[i] - y[i]) ** 2
    return s / len(X)

X = [1., 2., 3.]
Y = [1., 2., 3.]
count=1
W_val, cost_val = [], []
for i in range(-30, 51):
    W = i*0.1
    c = cost(W, X, Y)
    
    print(count, '{:.1f}, {:.1f}'.format(W, c))    
    count+=1
    W_val.append(W)
    cost_val.append(c)

# ------------------------------------------ #
    import matplotlib.pyplot as plt

    plt.plot(W_val, cost_val, 'ro')
    plt.ylabel('Cost')
    plt.xlabel('W')
    plt.show()
