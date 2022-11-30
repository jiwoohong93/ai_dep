import argparse
import os
from copy import deepcopy

from config import yaml_config_hook
from sipm.runner import _runner
from sipm.misc import save_result

if __name__ == '__main__':
    
    ''' parsers '''
    parser = argparse.ArgumentParser(description='Learning Fair Representation via sIPM')
    config = yaml_config_hook('config.yaml')
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=type(v))
    args = parser.parse_args()
    print(args)
        
    ''' running experiments '''
    Runner = _runner(args.dataset, args.scaling,
                     args.batch_size,
                     args.epochs, args.opt, args.model_lr, args.aud_lr, 
                     args.aud_steps, args.acti, args.num_layer, args.head_net, args.aud_dim,
                     args.eval_freq
                    )
    seeds = [2021, 2022, 2023, 2024, 2025] if bool(args.run_five) else [2021]
    stats = {}
    for i, seed in enumerate(seeds):
        print(f'::: STEP {i+1} with seed {seed} :::')
        # training
        Runner.learning(i, seed, args.lmda, args.lmdaF, args.lmdaR)
        print(Runner.results_path)
        # check whether already done
        if os.path.exists(Runner.results_path + 'mean_result.json'):
            break
        # inference     
        stat = Runner.inference(when='best')    
        if i == 0:
            stats = deepcopy(stat)
            for key in stats.keys():
                stats[key] = [stats[key]]
        else:
            for key in stats.keys():
                stats[key].append(stat[key])
    
    ''' saving results '''
    save_result(Runner.results_path, stats)
    print('DONE!')
