import random

import numpy as np
import matplotlib.pyplot as plt

k = 10 #number of arms
steps = 10000 #number of steps
trials = 2000 #number of trials

def sampling(mu=0, std=1.0, nb_samples=None):
    return np.random.normal(loc=mu, scale=std, size=nb_samples)

class Bandit:

    def __init__(self, k, epsilon, update_method, alpha=0.1):
        methods = ["sample_mean", "step_size_mean"]
        assert update_method in methods, "method {} is not implemented."
        if update_method == "sample_mean":
            self.update_Q = self.sample_mean
        elif update_method == "step_size_mean":
            self.update_Q = self.step_size_mean

        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha # step size parameter
        self.indices = np.arange(self.k)
        self.q = np.zeros(self.k) # true value
        self.Q = np.zeros(self.k) # estimated value
        self.N = np.zeros(self.k)
        self.selected_ind = None
        self.reward = None

    def step(self):
        ind = self.epsilon_greedy() # action
        value = self.sample_value(ind)
        self.N[ind] += 1
        self.update_Q(ind, value)
        self.step_q()
        self.selected_ind = ind
        self.reward = value

    def sample_value(self, ind):
        return sampling(mu=self.q[ind])

    def sample_mean(self, ind, value):
        self.Q[ind] = self.Q[ind] + (value - self.Q[ind]) / self.N[ind]

    def step_size_mean(self, ind, value):
        self.Q[ind] = self.Q[ind] + self.alpha * (value - self.Q[ind])

    def epsilon_greedy(self):
        e = np.random.rand()
        if e < self.epsilon:
            ind = random.choice(self.indices)
        else:
            ind = np.argmax(self.Q)
            if sum(self.Q == self.Q[ind]) > 1:
                ind = random.choice(self.indices[self.Q == self.Q[ind]])
        return ind

    def greedy(self):
        return np.argmax(self.Q)

    def step_q(self):
        self.q += sampling(std=0.01, nb_samples=10)

def plot_acc(sm_acc, ss_acc):
    plt.figure()
    plt.ylabel("Optimal Action")
    plt.xlabel("Steps")
    plt.plot(sm_acc*100, label="sample mean")
    plt.plot(ss_acc*100, label="step size mean")
    plt.legend()
    plt.savefig("accuracy_vs_steps.png")

def plot_reward(sm_r, ss_r):
    plt.figure()
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.plot(sm_r, label="sample mean")
    plt.plot(ss_r, label="step size mean")
    plt.legend()
    plt.savefig("reward_vs_steps.png")

def main():
    epsilon = 0.1

    # sample mean
    print("sample mean")
    actions = np.zeros((trials, steps))
    trues = np.zeros((trials, steps))
    rewards = np.zeros((trials, steps))
    for t in range(trials):
        if t % 25 == 0:
            print("trial: {}".format(t))
            
        p = Bandit(k, epsilon, "sample_mean")
        for i in range(steps):
            p.step()
            true = np.argmax(p.q)
            if sum(p.q == p.q[true]) > 1:
                true = random.choice(p.indices[p.q == p.q[true]])
            actions[t, i] = p.selected_ind
            trues[t, i] = true
            rewards[t, i] = p.reward
    sample_mean_acc = np.mean(actions == trues, axis=0)
    sample_mean_reward = np.mean(rewards, axis=0)
    
    # step size mean
    print("step size mean")
    actions = np.zeros((trials, steps))
    trues = np.zeros((trials, steps))
    for t in range(trials):
        if t % 25 == 0:
            print("trial: {}".format(t))
            
        p = Bandit(k, epsilon, "step_size_mean")
        for i in range(steps):
            p.step()
            true = np.argmax(p.q)
            if sum(p.q == p.q[true]) > 1:
                true = random.choice(p.indices[p.q == p.q[true]])
            actions[t, i] = p.selected_ind
            trues[t, i] = true
            rewards[t, i] = p.reward
    step_size_mean_acc = np.mean(actions == trues, axis=0)
    step_size_mean_reward = np.mean(rewards, axis=0)

    plot_acc(sample_mean_acc, step_size_mean_acc)
    plot_reward(sample_mean_reward, step_size_mean_reward)

if __name__ == "__main__":
    main()