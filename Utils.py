# Import necessary modules
import copy
import torch
import yaml

from Policy import Policy
from Individual import Individual

class Utils:
    def __init__(self, num_objs=2):
        self.num_objs = num_objs

    def SBX(self, x1: Policy, x2: Policy, eta=15):
        """
        Perform Simulated Binary Crossover (SBX) on two Policy instances.

        Parameters:
        - x1 (Policy): First parent Policy instance.
        - x2 (Policy): Second parent Policy instance.
        - eta (int, optional): Distribution index controlling the spread of offspring. Default is 15.

        Returns:
        - y1 (Policy): First offspring Policy instance.
        - y2 (Policy): Second offspring Policy instance.
        """
        # Create deep copies of the parent policies to serve as offspring
        y1 = copy.deepcopy(x1)
        y2 = copy.deepcopy(x2)

        with torch.no_grad():
            # Iterate over parameters (weights and biases) of the policies
            for param_x1, param_x2, param_y1, param_y2 in zip(
                x1.parameters(), x2.parameters(), y1.parameters(), y2.parameters()
            ):
                p1 = param_x1.data
                p2 = param_x2.data

                # Generate random numbers u between 0 and 1
                u = torch.rand_like(p1)

                # Compute beta_q using the SBX formula
                beta_q = torch.where(
                    u <= 0.5,
                    (2 * u) ** (1.0 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1.0 / (eta + 1)),
                )

                # Generate offspring parameters
                child1 = 0.5 * ((1 + beta_q) * p1 + (1 - beta_q) * p2)
                child2 = 0.5 * ((1 - beta_q) * p1 + (1 + beta_q) * p2)

                # Assign the new parameters to the offspring policies
                param_y1.data.copy_(child1)
                param_y2.data.copy_(child2)

        return y1, y2

    def crossover(self, parent1: Individual, parent2: Individual, glob_id_counter: int):
        """
        Perform a crossover between 2 Individuals.

        Parameters:
        - parent1 (Individual): First parent Individual instance.
        - parent2 (Individual): Second parent Individual instance.

        Returns:
        - offspring1 (Individual): First offspring Individual instance.
        - offspring2 (Individual): Second offspring Individual instance.
        """
        # Blank joint policies for the offsprings
        joint_policy1 = []
        joint_policy2 = []

        if len(parent1.joint_policy) != len(parent2.joint_policy):
            raise ValueError("Parents must have joint policies of equal length for crossover!")
        
        # Iterate through both parents' policies and perform SBX
        for p1, p2 in zip(parent1.joint_policy, parent2.joint_policy):
            o1, o2 = self.SBX(p1, p2) # Get two offspring policies
            joint_policy1.append(o1)
            joint_policy2.append(o2)

        return Individual(joint_policy=joint_policy1, num_objs=self.num_objs, id=glob_id_counter+1), Individual(joint_policy=joint_policy2, num_objs=self.num_objs, id=glob_id_counter+2)