import numpy as np
import random
import matplotlib.pyplot as plt

def epsilon(n):
    return np.mat([random.gauss(0,sig_y) for i in range(n)]).T

def y_real(beta, n):
    x = []
    for j in range(n):
        x_l = [random.gauss(0,sig_x) for i in range(p + 1)]
        x_l[0] = 1.0
        x.append(x_l)
    x = np.mat(x)
    y = np.matmul(x, beta) + epsilon(n)
    return y, x

if __name__ == '__main__':
    n = 300
    p = 20
    p1 = int(p * 0.5)
    sig_x = 0.2
    sig_y = 3
    x0 = [1 if i == 0 else 0.05 for i in range(p + 1)]
    M = 1000
    beta = np.mat([1 if i <= p1 else 0 for i in range(p + 1)]).T
    y0_e = np.matmul(x0, beta)
    bia = [[] for i in range(p + 1)]
    y0_hats = [[] for i in range(p + 1)]
    for m in range(M):
        y, x = y_real(beta,  n)
        re = np.matmul(x.T,x).I
        beta_hat = np.matmul(np.matmul(re,x.T),y)
        beta_hat_k = beta_hat
        for k in range(p + 1):
            y0_hat = np.matmul(x0, beta_hat_k)
            bi = float(y0_hat - y0_e)
            bia[p - k].append(bi)
            y0_hats[p - k].append(float(y0_hat))
            beta_hat_k[p - k] = 0

    bias = []
    mses = []
    for k in range(p + 1):
        y0_hat_l = np.array(y0_hats[k])
        bis = np.array(bia[k])
        biask = float(y0_hat_l.mean() - y0_e)
        bias.append(biask * biask)
        mse = bis * bis
        mses.append(mse.mean())

    T = np.array([x for x in range(p+1)])
    l1 = np.array(bias)
    l3 = np.array(mses)
    l2 = l3 - l1
    plt.plot(T, l1)
    plt.plot(T, l2)
    plt.plot(T, l3)
    plt.show()
