import copy
import gym
import numpy as np
import torch
import time
import matplotlib.pyplot as plt


class Individual(object):

    def __init__(self):
        self._genotype = []
        self._fitness = 0

    def get_gene(self, i):
        return self._genotype[i]

    def set_gene(self, gene, i):
        self._genotype[i] = gene

    def get_genotype(self):
        return self._genotype

    def set_genotype(self, genotype):
        self._genotype = genotype

    def get_fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness


class Population(object):

    def __init__(self, size):
        self._size = size
        self._population = []

    def initialise_population(self, num_gene_each_idv):
        gene_pop = np.random.normal(0, 0.1, (self._size, num_gene_each_idv))
        for i in range(len(gene_pop)):
            self._population.append(Individual())
            self._population[i].set_genotype(gene_pop[i])

    def get_best_individual(self):
        target = np.argmax(self.get_pop_fitness())
        return self._population[target]

    def get_population(self):
        return self._population

    def get_pop_fitness(self):
        result = []
        for idx in range(self._size):
            result.append(self._population[idx].get_fitness())
        return result

    def get_max_fitness(self):
        best = self.get_best_individual()
        return best.get_fitness()

    def get_avg_fitness(self):
        result = self.get_pop_fitness()
        return np.average(result)

    def modify_population(self, elite, loser, i):
        self._population[:i] = elite
        self._population[i:] = loser


class Agent(object):

    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.num_genes = num_input * num_output + num_output
        self.weights = None
        self.bias = None

    def set_genes(self, gene):
        weight_idx = self.num_input * self.num_output
        bias_idx = self.num_input * self.num_output + self.num_output
        w = gene[0: weight_idx].reshape(self.num_output, self.num_input)
        b = gene[weight_idx: bias_idx].reshape(self.num_output, )
        self.weights = torch.from_numpy(w)
        self.bias = torch.from_numpy(b)

    def get_action(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
        forward = torch.mm(obs, self.weights.T.to(torch.float32)) + self.bias
        if forward > 0:
            return 1
        else:
            return 0


class Environment(object):

    def __init__(self, pop_size, score, agent):
        # Parameters
        self._pop_size = pop_size
        self._score = score  # target score 500
        self.agent = agent

        # Infrastructure
        self._env = gym.make('CartPole-v1')
        self._pop = Population(self._pop_size)
        self._pop.initialise_population(self.agent.num_genes)

        # Record helpers
        self.best_record = []
        self._generation = -1
        self._start = time.time()
        self._end = 0

    def _run_episode(self, render=False):
        total_reward = 0.0
        obs = self._env.reset()
        for i in range(self._score):
            if render: self._env.render()
            action = self.agent.get_action(obs)
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            if done: break
        return total_reward

    def _calc_fitness(self, pop):
        for i, indv in enumerate(pop.get_population()):
            self.agent.set_genes(indv.get_genotype())
            pop.get_population()[i].set_fitness(self._run_episode())
        return pop.get_pop_fitness()

    def _elitism(self, elite_num):
        index = np.zeros(self._pop_size, dtype=np.int_)
        index[::-1] = np.asarray(self._pop.get_pop_fitness(), dtype=np.int_).argsort()
        elite = []
        [elite.append(self._pop.get_population()[i]) for i in index]
        return elite[:elite_num], elite[elite_num:]

    def _mutation(self, loser_group, prob=0.2):

        for loser_idv in loser_group:
            original_idv = copy.deepcopy(loser_idv)
            new_idv = copy.deepcopy(loser_idv)
            roll = np.random.uniform()

            if roll < prob:
                for i in range(len(loser_idv.get_genotype())):
                    new_idv.set_gene(np.random.normal(0, 0.1, 1)[0], i)
                    # new_idv.set_gene(np.random.rand(), i)

                    self.agent.set_genes(original_idv.get_genotype())
                    org_fitness = self._run_episode()
                    self.agent.set_genes(new_idv.get_genotype())
                    new_fitness = self._run_episode()

                    if new_fitness > org_fitness:
                        loser_idv.set_genotype(new_idv.get_genotype())

        return loser_group

    def _crossover(self, elite_group, loser_group, prob=0.8):

        crossed_pop = []
        for a in range(len(loser_group)):
            good_one = np.random.choice(elite_group)
            bad_one = np.random.choice(loser_group)

        for i in range(len(bad_one.get_genotype())):
            roll = np.random.uniform()
            if roll < prob:
                bad_one.set_gene(good_one.get_genotype()[i], i)
            crossed_pop.append(bad_one)

        return crossed_pop

    def run(self, num_epoch, elite_num, cross_rate, mutate_rate):
        epoch = 0
        while self._pop.get_max_fitness() < self._score:
            self._generation += 1

            prev_pop_fitness = self._calc_fitness(self._pop)
            best_idv = self._pop.get_best_individual()
            best_fitness = best_idv.get_fitness()
            self.best_record.append(best_fitness)
            if epoch % 5 == 0:
                print('Generation %d  --  Max Fitness = %.3f -- Mean Fitness = %.3f'
                      % (self._generation, best_idv.get_fitness(), self._pop.get_avg_fitness()))
            if best_fitness >= self._score: break

            elite, loser = self._elitism(elite_num)
            altered_loser = self._crossover(elite, loser, prob=cross_rate)
            altered_loser = self._mutation(altered_loser, prob=mutate_rate)
            self._pop.modify_population(elite, altered_loser, elite_num)

            epoch += 1
            if epoch >= num_epoch: break

        self._end = time.time()
        print('---------COMPLETED---------')
        print('Best Fitness = %.3f, in Generation %d, Current Mean = %.3f'
              % (self._pop.get_max_fitness(), self._generation, self._pop.get_avg_fitness()))
        print('Best Genotype is: ', self._pop.get_best_individual().get_genotype())
        print('Time Spent = %.3f sec' % (self._end - self._start))
        return self._pop.get_best_individual()


if __name__ == '__main__':
    pop_size = 15
    num_obs, num_actions = 4, 1

    # Training
    agent = Agent(num_obs, num_actions)
    model = Environment(pop_size, 500, agent)
    best_individual = model.run(num_epoch=1000, elite_num=10, cross_rate=0.9, mutate_rate=0.8)

    # Visualise the Best Individual
    record_obs = [[], [], [], []]
    agent = model.agent
    agent.set_genes(best_individual.get_genotype())
    obs = model._env.reset()

    for i in range(500):
        model._env.render()
        action = agent.get_action(obs)
        obs, reward, done, info = model._env.step(action)
        [record_obs[i].append(obs[i]) for i in range(len(obs))]
        if done: break

    plt.plot(model.best_record)
    plt.title('Best Fitness vs Tournaments')
    plt.xlabel('Tournaments')
    plt.ylabel('Best Fitness')
    plt.show()

    plt.plot(record_obs[0])
    plt.title('Cart Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Position of Cart')
    plt.yticks(np.arange(-2.4, 2.5, 0.6))
    plt.show()

    plt.plot(record_obs[1])
    plt.title('Cart Velocity vs Time')
    plt.xlabel('Time')
    plt.ylabel('Velocity of Cart')
    plt.show()

    plt.plot(record_obs[2])
    plt.title('Pole Angle vs Time')
    plt.xlabel('Time')
    plt.ylabel('Angle of Pole')
    plt.show()

    plt.plot(record_obs[3])
    plt.title('Rotation Rate vs Time')
    plt.xlabel('Time')
    plt.ylabel('Rotation Rate of Pole')
    plt.show()

