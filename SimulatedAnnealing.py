import time
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
# ------------------------------------------------------------------------------
# Customization section:


class SimulatedAnnealing:

    def __init__(self, initial_temperature=100, cooling=0.8, number_variables=2, upper_bounds=[3, 3],
                 lower_bounds=[-3, -3], computing_time=1, no_attempts=100, lr=True):
        self.initial_temperature = initial_temperature
        self.cooling = cooling  # cooling coefficient
        self.number_variables = number_variables
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.computing_time = computing_time  # second(s)
        self.no_attempts = no_attempts
        self.record_best_fitness = []
        self.record_best_fitness_acc = []
        self.plus = True
        self.lr = lr

    def save_model(self, nome, model):
        np.savetxt(nome + '_weights.txt', model['weights'], delimiter=',')
        np.savetxt(nome + '_bias.txt', model['bias'], delimiter=',')

    def objective_function(self, X, ob, model, data, r):
        model, loss, acc = ob(X, model, data, r)
        return model, loss, acc

    # ------------------------------------------------------------------------------
    # Simulated Annealing Algorithm:
    def initialize(self, epoch, obj, model, data):
        initial_solution = np.zeros(self.number_variables)
        for v in range(self.number_variables):
            if v == 0 and self.lr is True:
                initial_solution[v] = random.uniform(self.lower_bounds[v], self.upper_bounds[v])
            else:
                initial_solution[v] = int(random.randrange(self.lower_bounds[v], self.upper_bounds[v]))
        current_solution = initial_solution
        best_solution = initial_solution
        n = 1  # no of solutions accepted
        model, best_fitness, acc = self.objective_function(best_solution, obj, model, 0, data) # Melhor perda
        current_temperature = self.initial_temperature  # current temperature
        start = time.time()
          # number of attempts in each level of temperature

        for i in range(1, epoch):
            for j in range(self.no_attempts):
                for k in range(self.number_variables):
                    if self.plus:
                        if k == 0:
                            current_solution[k] = best_solution[k] + 0.1 * \
                                              (random.uniform(self.lower_bounds[k], self.upper_bounds[k]))
                        else:
                            current_solution[k] = best_solution[k] + 0.1 * \
                                              (random.randrange(self.lower_bounds[k], self.upper_bounds[k]))
                        current_solution[k] = max(min(current_solution[k], self.upper_bounds[k]),
                                                      self.lower_bounds[k])  # repair the solution respecting the bounds
                    else:
                        if k == 0:
                            current_solution[k] = best_solution[k] - 0.1 * \
                                              (random.uniform(self.lower_bounds[k], self.upper_bounds[k]))
                        else:
                            current_solution[k] = best_solution[k] - 0.1 * \
                                              (random.randrange(self.lower_bounds[k], self.upper_bounds[k]))
                        current_solution[k] = max(min(current_solution[k], self.upper_bounds[k]),
                                                      self.lower_bounds[k])  # repair the solution respecting the bounds
                model, current_fitness, acc = self.objective_function(current_solution, obj, model, i, data)
                energy = abs(current_fitness - best_fitness)
                #print(current_solution)
                if i == 1 and j == 0:
                    EA = energy

                if current_fitness > best_fitness:
                    self.plus = not self.plus
                    p = math.exp(-energy.numpy() / (EA * current_temperature))
                    # make a decision to accept the worse solution or not
                    if random.random() < p:
                        accept = True  # this worse solution is accepted
                    else:
                        accept = False  # this worse solution is not accepted
                else:
                    accept = True  # accept better solution
                if accept:
                    best_solution = current_solution  # update the best solution
                    model, best_fitness, acc = self.objective_function(best_solution, obj, model, i, data)
                    n = n + 1  # count the solutions accepted
                    EA = (EA * (n - 1) + energy) / n  # update EA

            #print('interation: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
            self.record_best_fitness.append(best_fitness.numpy())
            self.record_best_fitness_acc.append(acc)
            # Cooling the temperature
            current_temperature = current_temperature * self.cooling
            # Stop by computing time
            end = time.time()
            if end - start >= self.computing_time:
                pass
                #break
        #self.save_model('proposta', model)


    def plot(self):
        plt.plot(self.record_best_fitness)
        plt.show()

    def save(self, name="name"):

        f = open(name + '_loss.txt', 'a')
        f.write(str(self.record_best_fitness)[1:-1])
        f.close()
        f = open(name + '_accuracy.txt', 'a')
        f.write(str(self.record_best_fitness_acc)[1:-1])
        f.close()


if __name__ == '__main__':
    def obj(X, m, d):
        x = X[0]
        y = X[1]
        value = 3 * (1 - x) ** 2 * math.exp(-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(
            -x ** 2 - y ** 2) - 1 / 3 * math.exp(-(x + 1) ** 2 - y ** 2)
        return m, value

    sa = SimulatedAnnealing()
    sa.initialize(9999999, obj=obj, model='m', federated_train_data='d')
    sa.plot()