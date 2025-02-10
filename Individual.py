import yaml

import Policy

class Individual:
    def __init__(self,
                 joint_policy=None,
                 config_filename=None, 
                 num_agents=10, 
                 input_size=10, 
                 output_size=2, 
                 id=-1, 
                 num_objs=2):
        
        if joint_policy is None:
            self.joint_policy = [Policy.Policy(config_filename, input_size=input_size, output_size=output_size) for _ in range(num_agents)]
        else:
            self.joint_policy = joint_policy
        self.id = id
        self.num_objs = num_objs

        self.trajectory = None

        self.reset_fitness()
    
    def reset_fitness(self):
        """Zero the fitness of the individual."""
        self.fitness = [-1 for _ in range(self.num_objs)]
    
    def mutate(self):
        """Mutate each policy in the joint policy"""
        for p in self.joint_policy:
            p.mutate()
    
    def __str__(self):
        """Define the string representation of the Individual."""
        return f"Individual(id={self.id}, fitness={self.fitness})"
    
    def __repr__(self):
        """Define the representation of the Individual."""
        return self.__str__()