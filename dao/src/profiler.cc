
#include "DAO/profiler.h"
#include <cuda_runtime.h>
#include <assert.h>
namespace DAO {

bool profile_enabled = false; 
Profiler profiler;

Profiler& Profiler::set_tensors(const dynet::Tensor* t) {
    kernel.push_back(t); 
    tensor_sizes[t] = t->d.size() * sizeof(float);
    return *this;
}

Profiler& Profiler::set_tensors(const std::vector<dynet::Tensor*>& t) {
    for (auto& ptr: t) {
        set_tensors(ptr);
    }
    return *this;
}

Profiler& Profiler::set_tensors(const std::vector<const dynet::Tensor*>& t) {
    for (auto& ptr: t) {
        set_tensors(ptr);
    }
    return *this;
}

void Profiler::start() {
    cudaEventCreate(&start_event); 
    cudaEventRecord(start_event); 
}

void Profiler::stop() {
    cudaEvent_t stop; 
    cudaEventCreate(&stop); 
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 
    float milliseconds = 0; 
    cudaEventElapsedTime(&milliseconds, start_event, stop); 
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop);
    kernels.push_back({milliseconds, std::move(kernel)}); 
    assert(kernel.size() == 0);
}

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
void Profiler::dump(std::ostream& os) {
    os << tensor_sizes.size() << " " << kernels.size() << std::endl; 
    std::unordered_map<const dynet::Tensor*, size_t> tensor2id;
    for (auto& t: tensor_sizes) {
        tensor2id[t.first] = tensor2id.size();
        os << t.second << std::endl;
    }
    for (auto& k: kernels) {
        os << k.first << " ";
        os << k.second.size() << " "; 
        for (auto& t: k.second) {
            os << tensor2id[t] << " "; 
        }
        os << std::endl; 
    }
}

void Profiler::dump(const std::string& filename) {
    std::string filename_ = filename + ".trace";
    std::ofstream ofs(filename_.c_str());
    dump(ofs);
    ofs.close();
}

} //DAO