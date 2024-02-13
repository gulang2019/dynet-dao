#ifndef DAO_ENGINE_H 
#define DAO_ENGINE_H

#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/training.h>
#include <dynet/model.h>

#include <vector>

#include <DAO/allocator.h>
#include <DAO/utils.h>

namespace DAO {

void initialize();
void finalize();

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
    /**
     * \brief Constructor
     * \brief If training, it should be initialized after the Parameters are 
     * added to the ParameterCollection
     * \param trainer The trainer to be used in the engine;
    */
    Engine(dynet::Trainer* trainer = nullptr);
    ~Engine();
    /**
     * \brief Symbolic forward
     * \param cg The computation graph
     * \param expr The expression to be evaluated
     * \return The tensor handle of the expression
    */
    const dynet::Tensor& symbolic_forward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr);
    /**
     * \brief Symbolic backward
     * \param cg The computation graph
     * \param expr The loss expression
    */
    void symbolic_backward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr); 
    /**
     * \brief Symbolic backward
     * \param cg The computation graph
     * \param expr The loss expression
    */
    void symbolic_update();
    /** 
     * Run the instructions in the engine.
     * Keep the outputs of the forward pass in the engine.
     * Keep weights & optimizer states in the engine.
     * Drop the gradients & intermidiates.
     * \param max_gpu_mem The maximum GPU memory to be used
     * \param max_cpu_mem The maximum CPU memory to be used
    */
    void run(size_t* max_gpu_mem = nullptr, size_t* max_cpu_mem = nullptr);

    /** 
     * Clear up the all kernels, instructions, and outputs.
    */
    void reset();

    /**Utility functions*/
    void report(std::ostream& os = std::cout) const;
    void dump_statistics(const std::string& filename) const; 
    /**
     * \brief Return the vector value of the tensor
     * \param tensor The tensor to be evaluated; Must be the output of the forward pass
    */
    std::vector<float> as_vector(const dynet::Tensor& tensor);

private: 
    bool sanity_check();
    void run_forward(Instruction& inst);
    void run_backward(Instruction& inst);
    void run_update(Instruction& inst);
    struct FWDKernel {
        // dynet::VariableIndex fx_idx;
        dynet::Tensor* tmp_tensor = nullptr;
    };

    struct BWDKernel {
        unsigned ai;
        dynet::VariableIndex fx_idx;
    };

    struct UPDKernel {
        float gscale;
        std::vector<dynet::Tensor*> values;
        std::shared_ptr<dynet::ParameterStorage> p;
        std::shared_ptr<dynet::LookupParameterStorage> lp;
    };

    struct ACC_GRADKernel {
        dynet::ParameterNodeBase* pnode;
        dynet::Tensor* ndEdf;
        dynet::Tensor* grads;
        dynet::Tensor* tmp = nullptr;
    };

    /** newed variables */
    std::vector<std::vector<dynet::Tensor*>> nfxss;
    std::vector<std::vector<dynet::Tensor*>> fwdtmpss; 
    std::vector<std::vector<dynet::Tensor*>> ndEdfss;
    // the outputs of all forward pass; we would keep it until the next run happens
    std::unordered_set<dynet::Tensor*> outputs;

    /** kernels */
    std::vector<std::vector<FWDKernel> > fwd_kernelss;
    std::vector<std::vector<BWDKernel> > bwd_kernelss;
    std::vector<std::vector<UPDKernel> > upd_kernelss;
    std::vector<std::vector<ACC_GRADKernel>> acc_kernelss;

    // the number of forward pass that has been inplaced
    size_t n_fwd_inplaced = 0;
    // whether the engine has been evaluated
    bool evaluated = false; 

    std::vector<Instruction> instructions;
    
    dynet::Trainer* trainer; 
    std::unordered_set<void*> updated_params;

    Timer timer;
    Allocator& allocator;
    
};

} // namespace DAO 


#endif 