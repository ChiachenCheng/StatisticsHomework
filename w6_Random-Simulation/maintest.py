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

def getmean(l_all):
    l_all = np.array(l_all).T
    l = []
    for k in l_all:
        l.append(k.mean())
    return np.array(l)

if __name__ == '__main__':
    n = 300
    p = 20
    p1 = int(p * 0.5)
    sig_x = 0.2
    sig_y = 3
    M = 10000
    M1 = 100
    l1_all = []
    l2_all = []
    l3_all = []
    for m1 in range(M1):
        x0 = [1 if i == 0 else random.gauss(0.05,0.05) for i in range(p + 1)]
        beta = np.mat([random.gauss(1,1) if i <= p1 else 0 for i in range(p + 1)]).T
        y0_e = np.matmul(x0, beta)
        bia = [[] for i in range(p + 1)]
        y0_hats = [[] for i in range(p + 1)]
        for m in range(int(M/M1)):
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

        l1 = np.array(bias)
        l3 = np.array(mses)
        l2 = l3 - l1
        l1_all.append(l1)
        l2_all.append(l2)
        l3_all.append(l3)

    l1_f = getmean(l1_all)
    l2_f = getmean(l2_all)
    l3_f = getmean(l3_all)
    T = np.array([x for x in range(p + 1)])
    plt.plot(T, l1_f)
    plt.plot(T, l2_f)
    plt.plot(T, l3_f)
    plt.show()
