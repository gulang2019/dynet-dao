 # DAO simulator 

A simulator to investigate the memory access pattern of different machine learning workloads. 

## Overview of gpu memory access 
1. X: time, Y: size;
2. The distribution of different gpu memory access.
- Implementation: python 
  - Input: trace file; 
  - Output: the visualize figures; 


## Experiments

1. Different bandwidth's influence to the communicate pressure 
X: gpu memory size;
Y: compute time / h2d time / d2h time of different skip rate;
Implementation: 
    - Input: trace file; skip_rate; bandwidth; memory size;
    - Output: compute time, h2d time, d2h time; 
1. different heuristic policy for eviction and its influence to the performance and overhead; 
   1. allocation: BEST_FIT, First Fit;
   2. eviction:   Best Fit as a function of size and delay, first fit;

c++ trace name, skip rate, bandwidth, memory size, eviction policy, allocation policy -> compute time, h2d time, d2h time || OOM;
python trace file -> visualization; data collection -> analysis; 


bandwidth
```
device: 0
h2d bandwidth: 4.47418e+08 r2: 0.99609
d2h bandwidth: 4.41418e+08 r2: 0.997946
device: 1
h2d bandwidth: 1.78638e+08 r2: 0.999972
d2h bandwidth: 1.78401e+08 r2: 0.999991
```