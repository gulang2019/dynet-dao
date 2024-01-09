import matplotlib.pyplot as plt
import argparse 
import numpy as np 
import subprocess

parser = argparse.ArgumentParser(description='Visualize trace')
parser.add_argument('--filename', type=str, default='filename')
# /**
#  * n m 
#  * size1
#  * ...
#  * sizen
#  * t1 k1 a1 a2 ... ak1 
#  * t2 k2 a1 a2 ... ak2
#  * ...
#  * tm km a1 a2 ... akm
# */
from typing import List, Dict 
import multiprocessing 

class Kernel:
    size: int = 0
    time = 0.0 
    accesses: List[int] = []

class Trace:
    name: str = ''
    sizes: List[int] = []
    kernels: List[Kernel] = []

def read_trace(filename):
    trace = Trace()
    trace.name = filename.split('/')[-1].split('.')[0]
    with open(filename, 'r') as f:
        line = f.readline()
        n, m = map(int, line.split())
        for i in range(n):
            line = f.readline()
            trace.sizes.append(int(line)/1024)
        for i in range(m):
            line = f.readline()
            line = line.split()
            kernel = Kernel()
            kernel.time = float(line[0])
            kernel.accesses = list(map(int, line[2:]))
            kernel.size = sum([trace.sizes[x] for x in kernel.accesses])
            assert len(kernel.accesses) == int(line[1])
            trace.kernels.append(kernel)
    return trace

def visualize_trace(trace: Trace):
    print(f'number of kernels: {len(trace.kernels)}')
    print(f'number of sizes: {len(trace.sizes)}')
    fig, axes = plt.subplots(2, tight_layout=True)
    ax0, ax1 = axes 
    ax0.hist(trace.sizes, bins=100)
    ax0.set_title('sizes')
    x = list(range(len(trace.kernels)))
    y = np.array(list(x.size for x in trace.kernels))
    w = np.array(list(x.time for x in trace.kernels))

    # #plt.bar(x, height = y, width = w, color = colors, alpha = 0.8)

    # xticks = w.cumsum() - w/2
        
    # # w_new = [i/max(w) for i in w]
    # a = ax1.bar(xticks, height = y, width = w, alpha = 0.8)
    # _ = ax1.set_xticks(xticks, x)
    # ax1.set_xlabel('time')
    # ax1.set_ylabel('size')
    
    ax1.scatter(y, w, alpha=0.8)
    ax1.set_xlabel('size')
    ax1.set_ylabel('time')
    fig.savefig(f'{trace.name}.png')

def main(filename):
    trace = read_trace(filename)
    visualize_trace(trace)
    
def run_analysis(
    trace_file_name, 
    gpu_mem_limit, 
    mem_bandwidth, 
    allocation_strategy, 
    evict_strategy, 
    skip_rate,
    output_file_name 
):
    cmd = f"./build/dao-simulator {trace_file_name} {gpu_mem_limit} {mem_bandwidth} {allocation_strategy} {evict_strategy} {skip_rate} {output_file_name}"
    # print(cmd)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

from multiprocessing import Pool 

import os 
if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs('./output', exist_ok=True)
    processes = []
    arg_lists = []
    pool = Pool(processes=32)
    for model_size in ['124M', '355M', '774M']:
        for skip_rate in [0,0.1,0.2,0.3,0.4,0.5]:
            for gpu_mem_limit in [1,2,4,8,12,16]:
                for mem_bandwidth in [6.6, 13.1, 19.7, 26.2, 32.7, 39.3]:
                    for allocation_strategy in ['BEST_FIT']:
                        for evict_strategy in ['BELADY', 'WEIGHTED_BELADY', 'LRU', 'WEIGHTED_LRU']:
                            trace_file_name = f'/home/siyuanch/ssd/workspace_zelongg/dynet-dao/models/gpt2-{model_size}-{skip_rate}/train.trace'
                            if not os.path.exists(trace_file_name):
                                print(f'{trace_file_name} not exists')
                                continue
                            os.makedirs(f'./output/gpt2-{model_size}-{skip_rate}', exist_ok=True)
                            output_file_name = f'./output/gpt2-{model_size}-{skip_rate}/{gpu_mem_limit}-{mem_bandwidth}-{allocation_strategy}-{evict_strategy}.csv'
                            if os.path.exists(output_file_name):
                                continue
                            pool.apply_async(run_analysis, (trace_file_name, gpu_mem_limit, mem_bandwidth, allocation_strategy, evict_strategy, skip_rate, output_file_name))
                            # run_analysis(trace_file_name, gpu_mem_limit, mem_bandwidth, allocation_strategy, evict_strategy, 0.0, output_file_name)
    pool.close()
    pool.join()