#ifndef DAO_ENGINE_H 
#define DAO_ENGINE_H

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <vector>

#include <DAO/allocator.h>
#include <DAO/utils.h>

namespace DAO {

void initialize(int argc, char** argv);

struct Instruction{
    enum opcode_t{ 
        FORWARD,
        BACKWARD,
        UPDATE
    } opcode;
    std::shared_ptr<dynet::ComputationGraph> cg; 
    dynet::VariableIndex i;
    unsigned idx; 
};

struct Engine {
    Engine();

    const dynet::Tensor& symbolic_forward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr);
    void symbolic_backward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr); 
    void symbolic_update();

    void run();
    void run_forward(Instruction& inst);
    void run_backward(Instruction& inst);
    void run_update(Instruction& inst);
    void report(std::ostream& os = std::cout);
    std::vector<float> as_vector(const dynet::Tensor& tensor);

    struct FWDKernel {
        dynet::VariableIndex fx_idx;
        std::vector<dynet::VariableIndex> xs_idx;
        dynet::VariableIndex tmp_idx = (dynet::VariableIndex)(-1);
    };

    struct BWDKernel {
        unsigned ai;
        std::vector<dynet::VariableIndex> xs_idx;
        dynet::VariableIndex fx_idx;
        dynet::VariableIndex dEdfx_idx;
        dynet::VariableIndex dEdxai_idx;
    };

    std::vector<std::vector<dynet::Tensor*>> nfxss;
    std::vector<std::vector<dynet::Tensor*>> fwdtmpss; 
    std::vector<std::vector<dynet::Tensor*>> ndEdfss;

    std::unordered_set<dynet::Tensor*> outputs;

    std::vector<std::vector<FWDKernel> > fwd_kernelss;
    std::vector<std::vector<BWDKernel> > bwd_kernelss;

    std::vector<Instruction> instructions;
    // dynet::Trainer trainer; 
    Allocator& allocator;
    Timer timer;
};

} // namespace DAO 


#endif 