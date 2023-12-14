#pragma once 
#include <vector>
#include <unordered_map>
#include "globals.h"

struct Kernel {
    std::vector<TensorUID> accesses;
    std::vector<TensorUID> _free_list; 
    double time; 
};

struct Trace {
    std::vector<Kernel> kernels;
    std::unordered_map<TensorUID, size_t> tensor_sizes; 
}; 

struct Device {
    size_t mem_limit; 
    double mem_bandwidth; 
};

struct Algorithm {
};

struct Stream {
    double mem_bandwidth; 

    double time = 0;
    double copy(double start_time, size_t size) {
        return std::max(time, start_time) + size / mem_bandwidth;
    }
};
