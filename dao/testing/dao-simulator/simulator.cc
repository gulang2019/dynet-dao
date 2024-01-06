#include <cstddef>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <chrono>

#include "devices.h"
#include "kernel.h"
#include "allocator.h"
#include "memory_manager.h"

std::unordered_map<DoubleLinkedListStorage::allocation_strategy_t, std::string> allocation_strategy_names = {
    {DoubleLinkedListStorage::allocation_strategy_t::BEST_FIT, "BEST_FIT"},
    {DoubleLinkedListStorage::allocation_strategy_t::FIRST_FIT, "FIRST_FIT"}
};

std::unordered_map<std::string, DoubleLinkedListStorage::allocation_strategy_t> allocation_strategy_map = {
    {"BEST_FIT", DoubleLinkedListStorage::allocation_strategy_t::BEST_FIT},
    {"FIRST_FIT", DoubleLinkedListStorage::allocation_strategy_t::FIRST_FIT}
};

std::unordered_map<DoubleLinkedListStorage::eviction_strategy_t, std::string> eviction_strategy_names = {
    {DoubleLinkedListStorage::eviction_strategy_t::EVI_FIRST_FIT, "FIRST_FIT"},
    {DoubleLinkedListStorage::eviction_strategy_t::BELADY, "BELADY"},
    {DoubleLinkedListStorage::eviction_strategy_t::WEIGHTED_BELADY, "WEIGHTED_BELADY"},
    {DoubleLinkedListStorage::eviction_strategy_t::LRU, "LRU"},
    {DoubleLinkedListStorage::eviction_strategy_t::WEIGHTED_LRU, "WEIGHTED_LRU"}
};

std::unordered_map<std::string, DoubleLinkedListStorage::eviction_strategy_t> eviction_strategy_map = {
    {"FIRST_FIT", DoubleLinkedListStorage::eviction_strategy_t::EVI_FIRST_FIT},
    {"BELADY", DoubleLinkedListStorage::eviction_strategy_t::BELADY},
    {"WEIGHTED_BELADY", DoubleLinkedListStorage::eviction_strategy_t::WEIGHTED_BELADY},
    {"LRU", DoubleLinkedListStorage::eviction_strategy_t::LRU},
    {"WEIGHTED_LRU", DoubleLinkedListStorage::eviction_strategy_t::WEIGHTED_LRU}
};


void report(
    const std::string& trace_name, 
    Allocator& allocator,
    Trace& trace,
    Device& device,
    double skip_rate,
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy,
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy,
    std::chrono::microseconds::rep duration,
    std::ostream& o) {
    o << "key,value" << std::endl;
    o << "name," << trace_name << std::endl;
    o << "skip,"    << skip_rate << std::endl;
    o << "bandwidth," << device.mem_bandwidth << std::endl;
    o << "mem_limit," << device.mem_limit << std::endl;
    o << "allocation_strategy," << allocation_strategy_names[allocation_strategy] << std::endl;
    o << "eviction_strategy," << eviction_strategy_names[evict_strategy] << std::endl;
    o << "total_time," << allocator.get_total_time() << std::endl;
    o << "idle_rate," << allocator.get_idle_rate() << std::endl;
    o << "compute_time," << allocator.get_compute_time() << std::endl;
    o << "h2d_time," << allocator.get_h2d_time() << std::endl;
    o << "d2h_time," << allocator.get_d2h_time() << std::endl;
    o << "duration(ms)," << duration << std::endl;
}

real_time_t simulate(
    const std::string& trace_name, 
    Trace& trace, 
    Device& device,
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy, 
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy,
    double skip_rate, 
    const std::string& output_filename) {
    Allocator allocator(trace, device, allocation_strategy, evict_strategy); 
    // measure time 
    auto start = std::chrono::high_resolution_clock::now();

    auto & compute_stream = allocator.get_compute_stream();
    for (auto& kernel: trace.kernels) {
        allocator.prepare(kernel);
        compute_stream.commit(kernel.time);
        allocator.complete(kernel);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
    report (trace_name, allocator, trace, device, skip_rate, allocation_strategy, evict_strategy, duration.count(), std::cout);
    std::ofstream file(output_filename);
    report (trace_name, allocator, trace, device, skip_rate, allocation_strategy, evict_strategy, duration.count(), file);
    file.close();
    return allocator.get_total_time(); 
}

int verbose = 1; 
/**
 * n m 
 * size1
 * ...
 * sizen
 * t1 k1 a1 a2 ... ak1 
 * t2 k2 a1 a2 ... ak2
 * ...
 * tm km a1 a2 ... akm
*/
#define ROUND_UP(x, y) (((x) + (y) - 1) / (y) * (y))
Trace read_trace(std::string filename, double skip_rate = 0) {
    std::ifstream file(filename);
    Trace ret;
    int n, m;
    file >> n >> m;
    for (int i = 0; i < n; i++) {
        size_t size;
        file >> size;
        ret.tensor_sizes[i] = size;
    }
    size_t minimum_mem_requirement = 0;
    for (int i = 0; i < m; i++) {
        if (rand() / (double)RAND_MAX < skip_rate) {
            continue;
        }
        Kernel kernel;
        file >> kernel.time; 
        int k;
        file >> k; 
        size_t acc = 0;
        for (int j = 0; j < k; j++) {
            int a;
            file >> a;
            kernel.accesses.push_back(a);
            acc += ROUND_UP(ret.tensor_sizes[a], 256);
        }
        minimum_mem_requirement = std::max(minimum_mem_requirement, acc);
        ret.kernels.push_back(std::move(kernel));
    }
    DAO_INFO("minimum memory requirement: %lu", minimum_mem_requirement);
    file.close();
    return std::move(ret); 
}

int main(int argc, char** argv) {
    if(argc < 5) {
        std::cout << "Usage: " << argv[0] << " trace_file_name gpu_mem_limit[GB] mem_bandwidth[MB/s] allocation_strategy \
            evict_strategy skip_rate output_file_name" << std::endl;
        return 1;
    }
    std::string trace_name = std::string(argv[1]);
    Device device;
    device.mem_limit = std::stold(argv[2]) * (1 << 30); // kB 
    device.mem_bandwidth = std::stof(argv[3]) * (1 << 20); // kB/s
    std::cout << "device: " << device.mem_limit << " B, " << device.mem_bandwidth << " kB/s" << std::endl;
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy = allocation_strategy_map[argv[4]];
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy = eviction_strategy_map[argv[5]];
    double skip_rate = std::stof(argv[6]);
    Trace trace = read_trace(trace_name, skip_rate);
    std::cout << "trace: " << trace.kernels.size() << " kernels, " << trace.tensor_sizes.size() << " tensors" << std::endl;
    std::string output_filename = "train.csv";
    if (argc > 7) {
        output_filename = argv[7];
    }
    simulate(trace_name, trace, device, allocation_strategy, evict_strategy, skip_rate, output_filename);
    return 0;
}

