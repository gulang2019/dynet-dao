#include <DAO/DAO.h>
#include <dynet/dynet.h>
#include <memory>

using namespace DAO; 
using namespace std;
using namespace dynet; 

void test_eviction() {

    Allocator allocator(
        /*cpu_mem*/ 2048,
        /*gpu_mem*/ 2048,
        /*cpu_mem_limit*/ 4096,
        /*gpu_mem_limit*/ 0,
        /*cpu_grow_size*/ 1024,
        /*gpu_grow_size*/ 1024
    );
    DAO_INFO("created allocator");
    allocator.display(std::cout);
    Tensor a, b;
    a.d = {256};
    a.name = "a";
    b.d = {256};
    b.name = "b";
    {
        DAO::Kernel kernel;
        kernel.set_inputs(&a);
        kernel.set_outputs(&b);
        allocator.prepare(kernel);
        // do somthing 
        allocator.complete(kernel);
    }
    DAO_INFO("after first allocation");
    allocator.display(std::cout);

    // should evict a here; 
    Tensor c; 
    c.d = {256};
    c.name = "c";
    {
        DAO::Kernel kernel;
        kernel.set_inputs(&b);
        kernel.set_outputs(&c);
        allocator.prepare(kernel);
        allocator.complete(kernel);
    }

    DAO_INFO("after second allocation");
    allocator.display(std::cout);

    Tensor d;
    d.d = {256}; 
    d.name = "d";
    {
        DAO::Kernel kernel;
        kernel.set_inputs(&a);
        kernel.set_outputs(&d);
        allocator.prepare(kernel);
        allocator.complete(kernel);
    }

    DAO_INFO("after third allocation");
    allocator.display(std::cout);

    allocator.free(&a, &b, &c);
     DAO_INFO("after free");
    allocator.display(std::cout);
}


void test_overflow() {
    Allocator allocator(
        /*cpu_mem*/ 1024,
        /*gpu_mem*/ 4096,
        /*cpu_mem_limit*/ 1024,
        /*gpu_mem_limit*/ 1024,
        /*cpu_grow_size*/ 1024,
        /*gpu_grow_size*/ 1024
    );
    Tensor a, b;
    a.d = {1024};
    b.d = {256};
    DAO::Kernel kernel;
    kernel.set_inputs(&a);
    kernel.set_outputs(&b);
    allocator.display(std::cout);
    allocator.prepare(kernel);
    // do somthing 
    allocator.complete(kernel);
}

int main(){
    DAO::verbose = 1; 
    test_eviction();
    return 0;
}


