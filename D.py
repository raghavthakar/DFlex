import random
import numpy

import Algorithm
import Individual

class D(Algorithm.CoevolutionaryAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)

    def evolve(self, gen=0, traj_write_freq=100):
        print(gen)
        """Evolve the population using D."""
        # Shuffle each subpopulation
        for subpop in self.pop:
            random.shuffle(subpop)
        team_fitnesses = [] # To store the multibjective team fitness of each eval
        difference_evals = [[] for _ in range(self.team_size)] # To store the difference evals of each polcy (team_size*pop_size*num_objs)
        # Perform rollout and assign fitness to each team
        for eval_idx in range(self.pop_size):
            # Pick policies at the eval_index across all subpopulations
            team_policy = [self.pop[i][eval_idx] for i in range(len(self.pop))]
            # Condcut rollout
            trajectory, fitness_dict = self.interface.rollout(team_policy)
            self.glob_eval_counter += 1
            if len(fitness_dict) != self.num_objs:
                raise ValueError(f"[D] Expected {self.num_objs} objectives, but got {len(fitness_dict)}.")
            # Store fitness
            for f in fitness_dict:
                fitness_dict[f] = -fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
            team_fitnesses.append(fitness_dict)

            # Add this evaluation's data to the logger
            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=self.glob_eval_counter)
            self.data_logger.add_data(key='fitness', value=[fitness_dict[f] for f in fitness_dict])
            if gen == self.num_gens - 1 or gen % traj_write_freq == 0:
                self.data_logger.add_data(key='trajectory', value=trajectory)
            else:
                self.data_logger.add_data(key='trajectory', value=None)
            self.data_logger.write_data()

            # Counterfactual eval of each policy in this team policy
            for p_idx in range(len(team_policy)):
                # Trajectory with this policy's experience excluded
                cf_traj = [trajectory[i] for i in range(len(team_policy)) if i != p_idx]
                cf_fitness_dict = self.interface.evaluate_trajectory(cf_traj)
                for f in cf_fitness_dict:
                    cf_fitness_dict[f] = -cf_fitness_dict[f] # NOTE: The fitness sign is flipped to match Pygmo convention
                # Difference evalutions per-objective for this policy
                objectives = sorted(fitness_dict.keys())
                policy_d_vals = [fitness_dict[o] - cf_fitness_dict[o] for o in objectives]
                # Append to corresponding subpop d values
                difference_evals[p_idx].append(policy_d_vals)
                
        # Sort each subpop according to difference evaluations
        for subpop_idx, subpop_d_vals in enumerate(difference_evals):
            print(subpop_d_vals)
            sorted_indices = sorted(range(len(subpop_d_vals)), key=lambda i: subpop_d_vals[i][0])
            # Arrange the policies in subpop according to this sorted order
            sorted_subpop = []
            for policy_idx in sorted_indices:
                sorted_subpop.append(self.pop[subpop_idx][policy_idx])
            sorted_subpop = sorted_subpop[: self.pop_size // 2]
            # Keep only the top half of each subpop
            self.pop[subpop_idx] = sorted_subpop

        # Offspring creation in each subpop
        for subpop_idx, subpop in enumerate(self.pop):
            offspring_set = []
            # Fill up the offspring set to the pop_size via offspring-creation
            while len(subpop) + len(offspring_set) < self.pop_size:
                idx1, idx2 = random.sample(range(len(subpop)), 2)
                parent1 = subpop[min(idx1, idx2)] # choose the lower (more fit) option
                idx1, idx2 = random.sample(range(len(subpop)), 2)
                parent2 = subpop[min(idx1, idx2)] # choose the lower (more fit) option
                # Crossover the parent policies using SBX to get two offspring
                offspring1, offspring2 = self.utils.SBX(parent1, parent2)
                # Mutate the offsprings by adding noise
                offspring1.mutate()
                offspring2.mutate()
                offspring_set.extend([offspring1, offspring2])
            self.pop[subpop_idx].extend(offspring_set)