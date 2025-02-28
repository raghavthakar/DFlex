import random
import torch
import numpy
import sys
import datetime
import shutil # for file management

import D
import G

if __name__ == '__main__':
    assert len(sys.argv) == 9, "Correct usage: python alg_name domain_name data_dirpath alg_config env_config seed label traj_write_freq"
   
    # Process the command line args
    alg_name = sys.argv[1]
    assert alg_name in ['d', 'g'], "Unrecognised alg_name"
    domain_name = sys.argv[2]
    assert domain_name in ['rover'], 'Uncrecognised domain_name'
    data_dir = sys.argv[3]
    data_dir = data_dir+'/' if data_dir[-1]!='/' else data_dir # Add a directory '/' at the end
    src_alg_config_filename = sys.argv[4]
    src_env_config_filename = sys.argv[5]
    seed_val = int(sys.argv[6])
    seed_val_str = str(seed_val)
    label = sys.argv[7]
    traj_write_freq = int(sys.argv[8])

    # Datetime for file naming
    datetime_now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save data filename
    data_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_savedata.csv'
    # Create copy of configs at save data location
    dest_alg_config_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_algconfig.yaml'
    shutil.copyfile(src_alg_config_filename, dest_alg_config_filename)
    dest_env_config_filename = data_dir+alg_name+'_'+domain_name+'_'+seed_val_str+'_'+label+'_'+datetime_now_string+'_envconfig.yaml'
    shutil.copyfile(src_env_config_filename, dest_env_config_filename)

    # Set the seed value for all libraries
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    numpy.random.seed(seed_val)

    # Initialise the algorithm based on alg name
    if alg_name == 'd':
        alg = D.D(alg_config_filename=dest_alg_config_filename,
                                domain_name=domain_name,
                                data_filename=data_filename,
                                rover_config_filename=dest_env_config_filename)
    
    if alg_name == 'g':
        alg = G.G(alg_config_filename=dest_alg_config_filename,
                                domain_name=domain_name,
                                data_filename=data_filename,
                                rover_config_filename=dest_env_config_filename)
    
    # Run the algorithm
    for gen in range(alg.num_gens):
        alg.evolve(gen=gen, traj_write_freq=traj_write_freq)