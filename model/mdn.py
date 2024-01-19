import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt



# plt.figure(figsize=(8,8))
# plt.scatter(x_data,y_data,alpha=0.4)
# plt.scatter(x_test,y_pred,alpha=0.4,color='red')
# plt.show()


def mdn_loss_fn(y, mu, sigma, pi):
    # print(mu,sigma,pi)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)

class MDN(nn.Module):
    def __init__(self,n_hidden,n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1,n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
            # nn.LeakyReLU()
        )
        self.z_pi = nn.Linear(n_hidden,n_gaussians)
        self.z_mu = nn.Linear(n_hidden,n_gaussians)
        self.z_sigma = nn.Linear(n_hidden,n_gaussians)

    def forward(self,x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h),-1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))

        return pi, mu, sigma


if __name__ == "__main__":

    n_samples = 1000

    epsilon = torch.randn(n_samples)
    x_data = torch.linspace(-10, 10, n_samples)
    y_data = 7 * np.sin(0.75 * x_data) + 0.5 * x_data + epsilon

    x_max = x_data.max()
    y_max = y_data.max()
    x_data = x_data/x_max
    y_data = y_data/y_max

    y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)

    # plt.figure(figsize=(8,8))
    # plt.scatter(x_data,y_data,alpha=0.4)
    # plt.show()
    # exit()

    # n_input = 1
    # n_hidden = 20
    # n_ouput = 1
    #
    # model = nn.Sequential(nn.Linear(n_input, n_hidden),
    #                       nn.Tanh(),
    #                       nn.Linear(n_hidden, n_ouput))
    #
    # loss_fn = nn.MSELoss()
    # optimizer = torch.optim.RMSprop(model.parameters())
    #
    # for epoch in range(50000):
    #     y_pred = model(x_data)
    #     loss = loss_fn(y_pred, y_data)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if epoch % 1000 == 0:
    #         print(loss.data.tolist())

    n_samples = 200


    x_test = torch.linspace(-10, 10, n_samples).view(-1, 1)/x_max
    # y_pred = model(x_test).data


    model = MDN(n_hidden=40, n_gaussians=1)
    optimizer = torch.optim.Adam(model.parameters())





    for epoch in range(10000):
        # try:
            pi, mu, sigma = model(x_data)


            loss = mdn_loss_fn(y_data, mu, sigma, pi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 5000 == 0:
                print(loss.data.tolist())
        # except Exception as e:
        #     print("x_data:", x_data.view(1, -1))
        #     print("y_data:", y_data.view(1, -1))
        #     print(pi, mu, sigma)
        #     exit()

    pi, mu, sigma = model(x_test)
    # print("x_Test:",x_test)
    # print("res {}_{}_{}".format(str(pi),str(mu),str(sigma)))
    k = torch.multinomial(pi, 1).view(-1)
    # KKK = torch.multinomial(pi, 1)
    # print("res_K:",KKK.shape)
    # print("res_k",k)
    y_pred = torch.normal(mu, sigma)[np.arange(n_samples), k].data


    # print("mu.shape:",mu.shape)
    # print("data_mu:",mu)
    # print("sigma_shape:",sigma.shape)
    # print("data_sigma:",sigma)
    # print("[np.arange(n_samples), k]:",[np.arange(n_samples), k])
    # print("y_pred:",y_pred)


    plt.figure(figsize=(8, 8))
    plt.scatter(x_data, y_data, alpha=0.4)
    plt.scatter(x_test, y_pred, alpha=0.4, color='red')
    plt.show()
