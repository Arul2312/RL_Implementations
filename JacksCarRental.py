# Write a program for policy iteration and re-solve Jack's car rental problem with the following changes. One of Jack's
# employees at the first location rides a bus home each night and lives near the second location. She is happy to
# shuttle one car to the second location for free. Each additional car still costs $2, as do all cars moved in the other
# direction. In addition, Jack has limited parking space at each location. If more than 10 cars are kept overnight at a
# location (after any moving of cars), then an additional cost of $4 must be incurred to use a second parking lot
# (independent of how many cars are kept there). These sorts of nonlinearities and arbitrary dynamics often occur in
# real problems and cannot easily be handled by optimization methods other than dynamic programming. To check your
# program, first replicate the results given for the original problem. If your computer is too slow for the full
# problem, cut all the numbers of cars in half.

import numpy as np
import math

import matplotlib.pyplot as plt

class Params:
    def __init__(self):
        self.max_cars = 20
        self.max_moves = 5
        self.r_car = 10
        self.c_night = 4
        self.c_car = 2
        self.discount = 0.9
        self.request_first = 3
        self.request_second = 4
        self.return_first = 3
        self.return_second = 2
        self.theta = 0.01

class PolicyIteration:
    def __init__(self, params):
        self.params = Params()

        self.S = [(x,y) for x in range(self.params.max_cars + 1) for y in range(self.params.max_cars + 1)]
        self.V = np.zeros((self.params.max_cars + 1, self.params.max_cars + 1))
        self.pi = np.zeros((self.params.max_cars + 1, self.params.max_cars + 1))

        self.pif = []
    
    def solve(self):
        i = 0
        while True:
            print(f"Iteration: {i + 1}")

            self.pif.append(self.pi.copy())
            while True:
                delta = 0
                for s in self.S:
                    v = self.V[s]
                    self.V[s] = self.V_eval(s, self.pi[s])
                    delta = np.maximum(delta, abs(v - self.V[s]))
                if delta < self.params.theta:
                    break
                print(f"Delta: {delta}")

            
            policy_stable = True
            for s in self.S:
                old_action = self.pi[s]
                vals = {a : self.V_eval(s, a) for a in self.A(s)}
                self.pi[s] = np.random.choice([a for a, value in vals.items() if value == np.max(list(vals.values()))])
                if old_action != self.pi[s]:
                    policy_stable = False

            if policy_stable:
                break

            i += 1

    def A(self, s):
        vals = []
        A = [x for x in range(-self.params.max_moves, self.params.max_moves + 1)]
        s_first, s_second = s

        for a in A:
            if s_first - a < 0 or s_first - a > self.params.max_cars:
                continue
            if s_second + a < 0 or s_second - a > self.params.max_cars:
                continue
            vals.append(a)
            
        return vals
    
    def V_eval(self, s, a):
        val = 0
        s_first, s_second = s

        s_first -= int(a)
        s_second += int(a)

        cost = self.params.c_car * abs(a)

        sum_prob_i = 0
        for i in range(s_first + 1):
            if i == s_first:
                p_i = 1 - sum_prob_i
            else:
                p_i = PolicyIteration.poisson(self.params.request_first, i)
                sum_prob_i += p_i
            r_i = i * self.params.r_car
            sum_prob_j = 0
            for j in range(s_second + 1):
                if j == s_second:
                    p_j = 1 - sum_prob_j
                else:
                    p_j = PolicyIteration.poisson(self.params.request_second, j)
                    sum_prob_j += p_j
                r_j = j * self.params.r_car
                sum_prob_k = 0
                for k in range(self.params.max_cars + i - s_first + 1):
                    if k == self.params.max_cars + i - s_first:
                        p_k = 1 - sum_prob_k
                    else:
                        p_k = PolicyIteration.poisson(self.params.return_first, k)
                        sum_prob_k += p_k
                    sum_prob_l = 0
                    for l in range(self.params.max_cars + j - s_second + 1):
                        if l == self.params.max_cars + j - s_second:
                            p_l = 1 - sum_prob_l
                        else:
                            p_l = PolicyIteration.poisson(self.params.return_second, l)
                            sum_prob_l += p_l

                        val += p_i * p_j * p_k * p_l * (
                                 r_i + r_j - cost + self.params.discount * self.V[s_first - i + k, s_second - j + l])
        return val
    
    @staticmethod
    def poisson(lamb, n):
        """
        :param lamb: lambda parameter of poisson distribution, rate
        :param n: n variable of poisson distribution, number of occurrences
        :return: probability of the event
        """
        return (lamb ** n) * math.exp(-lamb) / math.factorial(n)
    
    def print_pif(self):
        """
        Print policies
        """
        for idx, pi in enumerate(self.pif):
            plt.figure()
            plt.imshow(pi, origin='lower', interpolation='none', vmin=-self.params.max_move, vmax=self.params.max_move)
            plt.xlabel('#Cars at second location')
            plt.ylabel('#Cars at first location')
            plt.title('pi{:d} {:s}')
            plt.colorbar()

    def print_V(self):
        """
        Print values
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, self.params.max_cars + 1)
        Y = np.arange(0, self.params.max_cars + 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, self.V)
        plt.title('V {:s}')

def main():
    param = Params()

    policy_iteration = PolicyIteration(param)
    policy_iteration.solve()

    policy_iteration.print_pif()
    policy_iteration.print_V()

    plt.show()

if __name__ == "__main__":
    main()