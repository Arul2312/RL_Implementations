# Consider driving a race car around a turn like those shown in Figure 5.5. You want to go as fast as possible, but not
# so fast as to run off the track. In our simplified racetrack, the car is at one of a discrete set of grid positions,
# the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally and vertically per
# time step. The actions are increments to the velocity components. Each may be changed by +1, -1, or 0 in each step,
# for a total of nine (3 ✕ 3) actions. Both velocity components are restricted to be nonnegative and less than 5, and
# they cannot both be zero except at the starting line. Each episode begins in one of the randomly selected start states
# with both velocity components zero and ends when the car crosses the finish line. The rewards are -1 for each step
# until the car crosses the finish line. If the car hits the track boundary, it is moved back to a random position on
# the starting line, both velocity components are reduced to zero, and the episode continues. Before updating the car's
# location at each time step, check to see if the projected path of the car intersects the track boundary. If it
# intersects the finish line, the episode ends, if it intersects anywhere else, the car is considered to have hit the
# track boundary and is sent back to the starting line. To make the task more challenging, with probability 0.1 at each
# time step the velocity increments are both zero, independently of the intended increments. Apply a Monte Carlo control
#  method to this task to compute the optimal policy from each starting state. Exhibit several trajectories following
# the optimal policy (but turn the noise off for these trajectories).

# The tracks were imported from another git library

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os
import csv

class Track:
    def __init__(self, track_type):
        
        self.STATE_OUT = 0
        self.STATE_IN = 1
        self.START = 2
        self.FINISH = 3

        self.track = []

        self.height = 0
        self.width = 0

        self.max_vel = 5
        self.min_vel = -5

        self.acc = 1

        self.actions = [[a_i, a_j] for a_j in range(-self.acc, self.acc + 1) for a_i in range(-self.acc, self.acc +1)]

        self.fail = 0.1

        self.RGB_BROWN = (139 / 255, 69 / 255, 19 / 255)
        self.RGB_GREEN = (.5, 1, 0)
        self.RGB_RED = (1, 0, 0)
        self.RGB_YELLOW = (1, 1, 0)
        self.RGB_BLACK = (0, 0, 0)

        assert track_type in ['type0', 'type1', 'type2', 'type3'], '\'{:s}\' not a valid name for track type'.format(
            track_type)
        script_dir = os.getcwd()
        absolute_path = os.path.join(script_dir, 'tracks', track_type, 'track.csv')
        
        self.load_track(absolute_path)


    def load_track(self, file_name):
        with open(file_name) as filen:
            self.track = [list(map(int, rec)) for rec in csv.reader(filen, delimiter=',')]

        self.height = len(self.track)
        self.width = len(self.track[0])

   
    def get_states(self, state_type):

        states = []

        for i in range(self.height):
            for j in range(self.width):
                if self.track[i][j] == state_type:
                    states.append([i,j])

        return states
    
    
    def take_action(self, state, action, is_exhibition):

        i, j, v_i, v_j = state
        a_i, a_j = action

        if np.random.binomial(1, self.fail) == 1 and not is_exhibition:
            a_i, a_j = 0, 0

        p_i = i
        p_j = j
        v_i += a_i
        v_j += a_j
        i -= v_i
        j += v_j

        done = False

        states_vis = [[i_, j_] for i_ in range(min(i, p_i), max(i, p_i) + 1) for j_ in range(min(j, p_j), max(j, p_j) + 1)]
        states_vis_type = [self.track[i_][j_] if 0 <= i_ < self.height and 0 <= j_ < self.width else self.STATE_OUT for i_, j_ in states_vis]

        if self.FINISH in states_vis_type:
            done = True

        elif self.STATE_OUT in states_vis_type:
            i, j, v_i, v_j = random.choice([[i, j, 0, 0] for i, j in self.get_states(self.START)])

        return [i, j, v_i, v_j], -1, done
    

    def A(self, state):

        actions = []
        
        A = self.actions.copy()
        _, _, v_i, v_j = state

        for a in A:
            a_i, a_j = a
            if v_i + a_i < self.min_vel or v_i + a_i > self.max_vel:
                continue
            if v_j + a_j < self.min_vel or v_j + a_j > self.max_vel:
                continue
            if v_i + a_i == 0 and v_j + a_j == 0:
                continue
            actions.append(a)
        return actions
    

    def print_track(self, state=None):
        tr_rgb = self.track.copy()
        tr_rgb = [[self.RGB_GREEN if s == self.STATE_OUT else self.RGB_BROWN if s == self.STATE_IN else
                  self.RGB_YELLOW if s == self.START else self.RGB_RED for s in row] for row in tr_rgb]
        if state is not None:
            x, y, _, _ = state
            tr_rgb[x][y] = self.RGB_BLACK
        im = plt.imshow(tr_rgb, origin='lower', interpolation='none', animated=True)
        plt.gca().invert_yaxis()
        return im
    


class Params:
    def __init__(self):

        self.discount = 1
        self.episodes = 100000

        self.exhibitions = 5

        self.start_q_value = -100000

        self.b_policy_type = 'epsilon_decay'

        if self.b_policy_type == 'epsilon_decay' or self.b_policy_type == 'random':
            self.epsilon = 1
        elif self.b_policy_type == 'deterministic':
            self.epsilon = 0
        else:
            self.epsilon = 0.01

        self.epsilon_decay, self.epsilon_min = 0.999, 0.1



class OffPolicyMC:
    def __init__(self, track, params):
        self.track = track

        self.params = params
        
        state_actions_shape = (self.track.height, self.track.width, self.track.max_vel - self.track.min_vel + 1, self.track.max_vel - self.track.min_vel + 1, 2 * self.track.acc + 1, 2 * self.track.acc + 1)

        states_shape = (self.track.height, self.track.width, self.track.max_vel - self.track.min_vel + 1, self.track.max_vel - self.track.min_vel + 1)

        self.Q = np.full(state_actions_shape, params.start_q_value)
        self.C = np.zeros(state_actions_shape)

        self.pi = np.empty(states_shape, dtype=object)
        for i in range(states_shape[0]):
            for j in range(states_shape[1]):
                for v_i in range(self.track.min_vel, self.track.max_vel + 1):
                    for v_j in range(self.track.min_vel, self.track.max_vel + 1):
                        self.pi[i, j, v_i, v_j] = random.choice(track.A([i, j, v_i, v_j]))


    def solve(self):

        i = 0

        while True:
            b = self.pi
            Ss, As, Rs = self.gen_episode(b, self.params.b_policy_type)

            G = 0
            W = 1

            print(f"Episode: {i}\t Step Needed: {len(Ss)}")

            for t in range(len(Ss) - 1, -1, -1):
                G = self.params.discount * G + Rs[t]
                self.C[tuple(Ss[t] + As[t])] += W
                self.Q[tuple(Ss[t] + As[t])] += (W / self.C[tuple(Ss[t] + As[t])]) * (G - self.Q[tuple(Ss[t] + As[t])])
                self.pi[tuple(Ss[t])] = random.choice([a for a in self.track.A(Ss[t]) if self.Q[tuple(Ss[t] + a)] == np.max([self.Q[tuple(Ss[t] + a)] for a in self.track.A(Ss[t])])])

                if As[t] != self.pi[tuple(Ss[t])]:
                    break
                W *= 1 / (1 - self.params.epsilon + self.params.epsilon/len(self.track.A(Ss[t])))

            i += 1
            if i > self.params.episodes:
                break

    
    def gen_episode(self, pi, policy_type, is_exhibition=False, S_0=None):
        assert S_0 is not None or not is_exhibition

        if policy_type == 'epsilon_decay' and self.params.epsilon > self.params.epsilon_min and not is_exhibition:
            self.params.epsilon *= self.params.epsilon_decay

        Ss = []
        As = []
        Rs = []

        if not is_exhibition:
            Ss.append(random.choice([[i, j, 0, 0] for i, j in self.track.get_states(self.track.START)]))
        else:
            Ss.append(S_0)

        index = 0
        while True:
            if policy_type == 'deterministic':
                
                As.append(pi[tuple(Ss[index])])
            elif np.random.binomial(1, self.params.epsilon) == 1 or policy_type == 'random':

                As.append(random.choice(self.track.A(Ss[index])))
            else:

                As.append(pi[tuple(Ss[index])])
            state, reward, done = self.track.take_action(Ss[index], As[index], is_exhibition)

            Rs.append(reward)
            if done:
                break
            else:
                Ss.append(state)
            index += 1
        return Ss, As, Rs
    
    def generate_exhibitions(self):

        starts = [[i, j, 0, 0] for i, j in self.track.get_states(self.track.START)]
        random.shuffle(starts)
        for idx, S_0 in enumerate(starts[:self.params.exhibitions]):

            Ss, As, Rs = self.gen_episode(self.pi, 'deterministic', is_exhibition=True, S_0=S_0)


            fig = plt.figure()
            ims = []
            for state in Ss:
                im = self.track.print_track(state)
                ims.append([im])
            anim = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=0)
            plt.show()



def main():
    track = Track('type1')

    params = Params()

    off_policy_monte_carlo = OffPolicyMC(track, params)

    off_policy_monte_carlo.solve()

    off_policy_monte_carlo.generate_exhibitions()


if __name__ == "__main__":
    main()
