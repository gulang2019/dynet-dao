#include <DAO/DAO.h>
#include <dynet/dynet.h>
#include <memory>

using namespace DAO; 
using namespace std;
using namespace dynet; 

logical_time_t logical_time;

void test_DoubleLinkedListStorage() {
    DAO_INFO("test_DoubleLinkedListStorage");
    std::unique_ptr<MemStorage> storage = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::CPU,
        1024*4,
        logical_time,
        size_t(-1),
        1024*1024,
        DoubleLinkedListStorage::BEST_FIT,
        DoubleLinkedListStorage::BEST_FIT,
        1024,
        128
    );
    DAO_INFO("created DoubleLinkedListStorage");
    storage->display(std::cout);
    GlobalMemRecord mem_record; 
    std::vector<Tensor> tensors;
    int n_tensor = 10;
    for (int i = 0; i < n_tensor; i++) {
        tensors.push_back(Tensor());
        tensors.back().d = {256 * (unsigned)((i+1) % 4)}; 
    }
    std::vector<std::shared_ptr<MemBlock> > blocks; 
    DAO_INFO("Allocate 10 blocks");
    for (int i = 0; i < n_tensor; i++) {
        auto& record = mem_record.lookup_tensor(&tensors[i]);
        record.access_pattern.push(0);
        record.access_pattern.push(i + 10); 
        DAO_INFO("Allocate %d blocks", i);
        blocks.push_back(storage->allocate(record.tensor_size));
        auto& block = blocks.back();
        block->allocated = true;
        block->record = &record; 
        record.block_ptr = block;
    }
    storage->display(std::cout);
    for (auto i: {0,1,2,3, 6,7,9}) {
        storage->free(blocks[i]);
        
    }
    DAO_INFO("After free");
    storage->display(std::cout);

    std::vector<std::shared_ptr<MemBlock> > evict_blocks; 
    std::shared_ptr<MemBlock> front, back; 
    std::tie(front, back) = storage->evict(4096, evict_blocks);
    assert(front != nullptr);
    for (auto& block: evict_blocks){
        block->allocated = false;
        block->record = nullptr;
    }
    DAO_INFO("Before merge");
    storage->display(std::cout);
    auto block = storage->mergeAndAllocate(4096, front, back);
    storage->display(std::cout);
}

void test_DoubleLinkedListStorageGPU() {
    DAO_INFO("test_DoubleLinkedListStorage");
    std::unique_ptr<MemStorage> storage = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::GPU,
        1024*4,
        logical_time,
        size_t(-1),
        1024*1024,
        DoubleLinkedListStorage::BEST_FIT,
        DoubleLinkedListStorage::BEST_FIT,
        1024,
        128
    );
    DAO_INFO("created DoubleLinkedListStorage");
    storage->display(std::cout);
    GlobalMemRecord mem_record; 
    std::vector<Tensor> tensors;
    int n_tensor = 10;
    for (int i = 0; i < n_tensor; i++) {
        tensors.push_back(Tensor());
        tensors.back().d = {256 * (unsigned)((i+1) % 4)}; 
    }
    std::vector<std::shared_ptr<MemBlock> > blocks; 
    DAO_INFO("Allocate 10 blocks");
    for (int i = 0; i < n_tensor; i++) {
        auto& record = mem_record.lookup_tensor(&tensors[i]);
        record.access_pattern.push(0);
        record.access_pattern.push(i + 10); 
        DAO_INFO("Allocate %d blocks", i);
        blocks.push_back(storage->allocate(record.tensor_size));
        auto& block = blocks.back();
        block->allocated = true;
        block->record = &record; 
        record.block_ptr = block;
    }
    storage->display(std::cout);
    for (auto i: {0,1,2,3, 6,7,9}) {
        storage->free(blocks[i]);
        
    }
    DAO_INFO("After free");
    storage->display(std::cout);

    std::vector<std::shared_ptr<MemBlock> > evict_blocks; 
    std::shared_ptr<MemBlock> front, back; 
    std::tie(front, back) = storage->evict(4096, evict_blocks);
    assert(front != nullptr);
    for (auto& block: evict_blocks){
        block->allocated = false;
        block->record = nullptr;
    }
    DAO_INFO("Before merge");
    storage->display(std::cout);
    auto block = storage->mergeAndAllocate(4096, front, back);
    storage->display(std::cout);
}


int main(){
    DAO::verbose = 1; 
    test_DoubleLinkedListStorage();
    test_DoubleLinkedListStorageGPU();
    return 0;
}