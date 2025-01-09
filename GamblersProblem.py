# Implement value iteration for the gambler's problem and solve it for ph = 0.25 and ph = 0.55. In programming, you may
# find it convenient to introduce two dummy states corresponding to termination with capital of 0 and 100, giving them
# values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3. Are your results stable as theta→0?

import numpy as np
import matplotlib.pyplot as plt

class Params:
    def __init__(self):
        self.max_money = 100
        self.ph = [0.25, 0.4, 0.5]
        self.theta = 10**-15
        self.discount = 1

class ValueIteration:
    def __init__(self, ph, params):
        self.params = params
        self.ph = ph
        self.S = np.arange(1, self.params.max_money)
        self.V = np.zeros(self.params.max_money + 1)
        self.V[0] = 0
        self.V[self.params.max_money] = 1
        self.Vs = []
        self.pi = None
        self.sweep = None

    def solve(self):
        self.sweep = 0
        while True:
            delta = 0
            for s in self.S:
                v = self.V[s]
                self.V[s] = np.max([self.V_eval(s, a) for a in self.A(s)])
                delta = np.maximum(delta, abs(v - self.V[s]))
            if self.sweep < 3:
                self.Vs.append(self.V.copy())
            self.sweep += 1
            if delta < self.params.theta:
                break
        print('Sweeps needed:', self.sweep)
        self.Vs.append(self.V.copy())
        self.pi = [self.A(s)[np.argmax([self.V_eval(s, a) for a in self.A(s)])] for s in self.S]

    def A(self, s):
        return np.arange(1, np.minimum(s, self.params.max_money - s) + 1)


    def V_eval(self, s, a):
        return self.params.discount * self.V[s + a] * self.ph + self.params.discount * self.V[s - a] * (1 - self.ph)

    def print_V(self):
        plt.figure()
        plt.plot(self.Vs[0], label='sweep 1')
        plt.plot(self.Vs[1], label='sweep 2')
        plt.plot(self.Vs[2], label='sweep 3')
        plt.plot(self.Vs[3], label='sweep {}'.format(self.sweep))
        plt.legend(loc='upper left')
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.title('Values ph={}'.format(self.ph))

    def print_pi(self):
        plt.figure()
        plt.step(self.S, self.pi)
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')
        plt.title('pi ph={}'.format(self.ph))


def main():
    params = Params()

    for ph in params.ph:
        policy_iteration = ValueIteration(ph, params)

        policy_iteration.solve()

        policy_iteration.print_V()
        policy_iteration.print_pi()


if __name__ == "__main__":
    main()
