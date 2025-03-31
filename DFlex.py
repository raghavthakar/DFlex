import random
import numpy

import Algorithm
import Individual

class DFlex(Algorithm.CoevolutionaryAlgorithm):
    def __init__(self, alg_config_filename, domain_name, rover_config_filename, data_filename):
        super().__init__(alg_config_filename, domain_name, rover_config_filename, data_filename)

    def evolve(self, gen=0, traj_write_freq=100):
        print(gen)
        """Evolve the population using D."""
        # Shuffle each subpopulation
        for subpop in self.pop:
            random.shuffle(subpop)
        # To store the team fitness from each rollout (not strictly required for D, but kept for consistency)
        team_fitnesses = []
        # difference_evals[p_idx] will store one or more difference vectors for policy p_idx
        difference_evals = [[] for _ in range(self.team_size)]

        # Perform rollout and assign fitness to each team
        for eval_idx in range(self.pop_size):
            # Pick policies at the eval_index across all subpopulations
            team_policy = [self.pop[i][eval_idx] for i in range(len(self.pop))]
            # Conduct rollout
            trajectory, fitness_dict = self.interface.rollout(team_policy)
            self.glob_eval_counter += 1
            if len(fitness_dict) != self.num_objs:
                raise ValueError(f"[D] Expected {self.num_objs} objectives, but got {len(fitness_dict)}.")
            # Flip sign to match Pygmo's minimization convention
            for f in fitness_dict:
                fitness_dict[f] = -fitness_dict[f]
            team_fitnesses.append(fitness_dict)

            # Log data
            self.data_logger.add_data(key='gen', value=gen)
            self.data_logger.add_data(key='id', value=self.glob_eval_counter)
            self.data_logger.add_data(key='fitness', value=[fitness_dict[f] for f in fitness_dict])
            if gen == self.num_gens - 1 or gen % traj_write_freq == 0:
                self.data_logger.add_data(key='trajectory', value=trajectory)
            else:
                self.data_logger.add_data(key='trajectory', value=None)
            self.data_logger.write_data()

            # Create a mapping from policy idx to credit set
            team_size = len(team_policy)
            credit_sets = [[] for _ in range(team_size)]
            if self.random_each_gen:
                raise ValueError("Random credit sets have been disabled!")
            else:
                if self.share_credit:
                    # Partition the team into equal chunks of size num_cf
                    if team_size % self.num_cf != 0:
                        raise ValueError("Team size must be a multiple of num_cf for shared credit.")
                    chunk_size = self.num_cf
                    for p_idx in range(team_size):
                        start = (p_idx // chunk_size) * chunk_size
                        credit_sets[p_idx] = list(range(start, start + chunk_size))
                else:
                    # Sliding window of width num_cf around each agent index
                    for p_idx in range(team_size):
                        half_left = self.num_cf // 2
                        offsets = range(-half_left, self.num_cf - half_left)
                        window = [(p_idx + off) % team_size for off in offsets]
                        credit_sets[p_idx] = window

            # Counterfactual evaluation
            if self.share_credit:
                # Gather unique sets (since multiple policies share the same chunk)
                unique_chunks = {}
                for p_idx, cset in enumerate(credit_sets):
                    # Sort so that the same chunk in different orders is recognized
                    key = tuple(sorted(cset))
                    if key not in unique_chunks:
                        unique_chunks[key] = []
                    unique_chunks[key].append(p_idx)

                # Evaluate once per unique chunk
                for cset_tuple, policy_indices in unique_chunks.items():
                    cset_list = list(cset_tuple)
                    # Trajectory without this chunk
                    cf_traj = [trajectory[i] for i in range(team_size) if i not in cset_list]
                    cf_fitness_dict = self.interface.evaluate_trajectory(cf_traj)
                    for f in cf_fitness_dict:
                        cf_fitness_dict[f] = -cf_fitness_dict[f]
                    objectives = sorted(fitness_dict.keys())
                    diff_vals = [fitness_dict[o] - cf_fitness_dict[o] for o in objectives]
                    # Assign same difference to each agent in the chunk
                    for p_idx in policy_indices:
                        difference_evals[p_idx].append(diff_vals)
            else:
                # Sliding window => each agent has its own credit set
                for p_idx, cset in enumerate(credit_sets):
                    cf_traj = [trajectory[i] for i in range(team_size) if i not in cset]
                    cf_fitness_dict = self.interface.evaluate_trajectory(cf_traj)
                    for f in cf_fitness_dict:
                        cf_fitness_dict[f] = -cf_fitness_dict[f]
                    objectives = sorted(fitness_dict.keys())
                    policy_d_vals = [fitness_dict[o] - cf_fitness_dict[o] for o in objectives]
                    difference_evals[p_idx].append(policy_d_vals)

        # Sort each subpop according to difference evaluations
        for subpop_idx, subpop_d_vals in enumerate(difference_evals):
            # This sorts purely by the first objective in each difference vector
            sorted_indices = sorted(range(len(subpop_d_vals)), key=lambda i: subpop_d_vals[i][0])
            sorted_subpop = []
            for policy_idx in sorted_indices:
                sorted_subpop.append(self.pop[subpop_idx][policy_idx])
            # Keep only the top half of each subpop
            self.pop[subpop_idx] = sorted_subpop[: self.pop_size // 2]

        # Offspring creation in each subpop
        for subpop_idx, subpop in enumerate(self.pop):
            offspring_set = []
            # Fill up the offspring set to pop_size
            while len(subpop) + len(offspring_set) < self.pop_size:
                idx1, idx2 = random.sample(range(len(subpop)), 2)
                parent1 = subpop[min(idx1, idx2)]
                idx1, idx2 = random.sample(range(len(subpop)), 2)
                parent2 = subpop[min(idx1, idx2)]
                # Crossover the parent policies using SBX to get two offspring
                offspring1, offspring2 = self.utils.SBX(parent1, parent2)
                # Mutate the offsprings by adding noise
                offspring1.mutate()
                offspring2.mutate()
                offspring_set.extend([offspring1, offspring2])
            self.pop[subpop_idx].extend(offspring_set)
