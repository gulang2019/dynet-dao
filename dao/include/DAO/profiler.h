#include "dynet/dynet.h"
#include "dynet/tensor.h"

#include <cuda_runtime.h>
#include <vector> 
#include <unordered_map>
#include <fstream>

namespace DAO {

struct Profiler {
    std::vector<const dynet::Tensor*> kernel;
    std::vector<std::pair<double, std::vector<const dynet::Tensor*>> > kernels;

    std::unordered_map<const dynet::Tensor*, size_t> tensor_sizes;  
    cudaEvent_t start_event; 
    Profiler& set_tensors(const dynet::Tensor* t); 
    Profiler& set_tensors(const std::vector<dynet::Tensor*>& t);
    Profiler& set_tensors(const std::vector<const dynet::Tensor*>& t);
    void start();
    void stop();
    void dump(std::ostream& os);
    void dump(const std::string& filename);
};

}