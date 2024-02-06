#include "DAO/engine.h" 
#include <dynet/sig.h>
#include <dynet/devices.h>
#include <dynet/param-nodes.h>
#include <dynet/globals.h>
#include <dynet/weight-decay.h>

#include <cuda_runtime.h>
#include <string>

namespace DAO { 

using dynet::Node;
using dynet::Tensor; 
using dynet::VariableIndex;

bool enabled = false;

std::unordered_map<int, const char *> nt2str = {
    {0, "unknown"},
    {dynet::nt::NodeType::tanh,"tanh"}, {dynet::nt::NodeType::sqrt,"sqrt"}, {dynet::nt::NodeType::abs,"abs"}, {dynet::nt::NodeType::erf,"erf"}, {dynet::nt::NodeType::square,"square"}, {dynet::nt::NodeType::cube,"cube"}, {dynet::nt::NodeType::exp,"exp"}, {dynet::nt::NodeType::logsigmoid,"logsigmoid"}, {dynet::nt::NodeType::loggamma,"loggamma"}, {dynet::nt::NodeType::log,"log"}, {dynet::nt::NodeType::nobackprop,"nobackprop"}, {dynet::nt::NodeType::scalegradient,"scalegradient"}, {dynet::nt::NodeType::identity,"identity"}, {dynet::nt::NodeType::negate,"negate"}, {dynet::nt::NodeType::rectify,"rectify"}, {dynet::nt::NodeType::logistic,"logistic"}, {dynet::nt::NodeType::softsign,"softsign"}, {dynet::nt::NodeType::silu,"silu"}, {dynet::nt::NodeType::round,"round"}, {dynet::nt::NodeType::ceiling,"ceiling"}, {dynet::nt::NodeType::floor,"floor"},
    {dynet::nt::NodeType::sinh,"sinh"}, {dynet::nt::NodeType::cosh,"cosh"}, {dynet::nt::NodeType::asinh,"asinh"}, {dynet::nt::NodeType::acosh,"acosh"}, {dynet::nt::NodeType::atanh,"atanh"}, {dynet::nt::NodeType::sin,"sin"}, {dynet::nt::NodeType::cos,"cos"}, {dynet::nt::NodeType::tan,"tan"}, {dynet::nt::NodeType::asin,"asin"}, {dynet::nt::NodeType::acos,"acos"}, {dynet::nt::NodeType::atan,"atan"}, {dynet::nt::NodeType::plus_const,"plus_const"}, {dynet::nt::NodeType::concat,"concat"}, {dynet::nt::NodeType::cmult,"cmult"}, {dynet::nt::NodeType::csum,"csum"}, {dynet::nt::NodeType::sum,"sum"}, {dynet::nt::NodeType::squared_distance,"squared_distance"}, {dynet::nt::NodeType::softmax,"softmax"}, {dynet::nt::NodeType::pnls,"pnls"}, {dynet::nt::NodeType::pickrange,"pickrange"}, {dynet::nt::NodeType::scalar_mult,"scalar_mult"}, {dynet::nt::NodeType::dropout,"dropout"},
    {dynet::nt::NodeType::input,"input"}, {dynet::nt::NodeType::scalar_input,"scalar_input"}, {dynet::nt::NodeType::lookup,"lookup"},
    {dynet::nt::NodeType::COMPLEX,"COMPLEX"},
    {dynet::nt::NodeType::affine,"affine"}, {dynet::nt::NodeType::matmul,"matmul"}, {dynet::nt::NodeType::transpose,"transpose"},
    {dynet::nt::NodeType::vanilla_lstm_gates,"vanilla_lstm_gates"}, {dynet::nt::NodeType::vanilla_lstm_h,"vanilla_lstm_h"}, {dynet::nt::NodeType::vanilla_lstm_c,"vanilla_lstm_c"},
    {dynet::nt::NodeType::conv2d,"conv2d"},
}; 

inline const char* node2str(const Node* node) {
    static dynet::SigMap sigmap;
    auto t = sigmap.sig2type(node->autobatch_sig(*node->get_cg(), sigmap));
    if (t == 0) return nullptr;
    return nt2str[t];
}



Engine::Engine(dynet::Trainer* trainer): allocator(dao_allocator), trainer(trainer) {
    if (trainer) {
        const auto & params = trainer->model->parameters_list();
        const auto & lparams = trainer->model->lookup_parameters_list();

        // Allocate if necessary
        if(trainer->aux_allocated < params.size()) {
            trainer->aux_allocated = trainer->alloc_impl();
        }
        if(trainer->aux_allocated_lookup < lparams.size()) {
            trainer->aux_allocated_lookup = trainer->alloc_lookup_impl();
        }
    }
}

std::vector<float> Engine::as_vector(const dynet::Tensor& tensor) {
    return std::move(allocator.get_values((TensorUID)&tensor));
}

bool Engine::sanity_check() {
    size_t n_kernels = 0;
    for (auto& fwd_kernelss: fwd_kernelss) {
        n_kernels += fwd_kernelss.size();
    }
    for (auto& bwd_kernelss: bwd_kernelss) {
        n_kernels += bwd_kernelss.size();
    }
    for (auto& upd_kernelss: upd_kernelss) {
        n_kernels += upd_kernelss.size();
    }
    for (auto& acc_kernels: acc_kernelss) {
        n_kernels += acc_kernels.size();
    }
    return n_kernels == (allocator.get_num_registered_kernels() + n_fwd_inplaced);
}

const Tensor& Engine::symbolic_forward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr) {
    timer.start("symbolic_forward");
    instructions.push_back({Instruction::FORWARD, cg, expr.i, (unsigned)nfxss.size()});
    nfxss.push_back({});
    fwdtmpss.push_back({});
    fwd_kernelss.push_back({});

    auto& nfxs = nfxss.back();
    auto& fwdtmps = fwdtmpss.back();
    auto& fwd_kernels = fwd_kernelss.back();
    fwd_kernels.resize(expr.i + 1, {});
    nfxs.resize(expr.i + 1, nullptr);

    dynet::VariableIndex param_idx = 0;
    for (dynet::VariableIndex num_nodes_evaluated = 0; 
        num_nodes_evaluated <= expr.i; ++num_nodes_evaluated) {
        DAO_INFO_LEVEL(2, "Symbolic FWD %u %s", num_nodes_evaluated, cg->nodes[num_nodes_evaluated]->as_dummy_string().c_str());
        const Node* node = cg->nodes[num_nodes_evaluated];
        auto& node_fx = nfxs[num_nodes_evaluated] = new dynet::Tensor();
        node_fx->d = node->dim;
        // Get the device
        DYNET_ASSERT(node->device != nullptr,
        "Attempt to access null device in "
        "SimpleExecutionEngine::incremental_forward");
        node_fx->device = node->device;
        node_fx->mem_pool = dynet::DeviceMempool::FXS;
        node_fx->name = node->as_dummy_string() + "_fx";

        // for inplaced node, we only need to copy the pointer
        if (node->forward_inplaced()) {
            DAO_ASSERT(!cg->nodes[node->args[0]]->forward_inplaced(), "cg->nodes[node->args[0]]->forward_inplaced()");
            n_fwd_inplaced ++;
            continue;
        }

        auto& fwd_kernel = fwd_kernels[num_nodes_evaluated];
        DAO::Kernel kernel;
        fwd_kernel.tmp_tensor = nullptr;
        if (node->aux_storage_size()) {
            fwd_kernel.tmp_tensor = new dynet::Tensor();
            fwd_kernel.tmp_tensor->name = node->as_dummy_string() + "_tmp";
            assert(fwd_kernel.tmp_tensor);
            fwd_kernel.tmp_tensor->d = dynet::Dim({(unsigned)(node->aux_storage_size() / sizeof(float) + 1)});
            fwdtmps.push_back(fwd_kernel.tmp_tensor);
            kernel.set_inputs(fwd_kernel.tmp_tensor);
        }
        std::vector<const dynet::Tensor*> xs;
        for (auto arg: node->args) {
            xs.push_back(nfxs[arg]);
            // if it is inplaced, we prepare the child node for the uniqueness; 
            if (cg->nodes[arg]->forward_inplaced()) {
                xs.back() = nfxs[cg->nodes[arg]->args[0]];
            }
        }
        kernel.set_inputs(xs).set_outputs(node_fx);

        // add the parameter input into the kernel
        if (param_idx < cg->parameter_nodes.size()
        && cg->parameter_nodes[param_idx] == num_nodes_evaluated) {
            auto param_base = static_cast<const dynet::ParameterNodeBase*>(node);
            if (param_base->type == dynet::ParameterNodeBase::PARAM) {
                auto param_node = static_cast<const dynet::ParameterNode*>(node);
                DAO_ASSERT((param_node->params.p == nullptr)^(param_node->lparams.p == nullptr), "param_node->params.p == nullptr)^(param_node->lparams == nullptr)");
                if(param_node->params.p != nullptr) // Parameter
                    kernel.set_inputs(&param_node->params.get_storage().values);
                else kernel.set_inputs(&param_node->lparams.get_storage().all_values);
            }
            else if (param_base->type == dynet::ParameterNodeBase::LOOKUP){
                auto lookup_node = static_cast<const dynet::LookupNode*>(node);
                DAO_ASSERT(lookup_node->params.p != nullptr, "lookup_node->params.p == nullptr");
                kernel.set_inputs(&lookup_node->params.get_storage().all_values);
            }
            else {DAO_ERROR("Unknown ParameterNodeBase type");}
            param_idx++;
        }

        allocator.Register(std::move(kernel));
    }
    allocator.set_record_type(nfxs.back(), DAO::TensorRecord::OUTPUT);
    timer.stop("symbolic_forward");
    DAO_ASSERT(sanity_check(), "sanity_check() failed");
    return *nfxs.back();
}

void Engine::symbolic_backward(std::shared_ptr<dynet::ComputationGraph> cg, 
            const dynet::Expression& expr) {
    timer.start("symbolic_backward");
    size_t num_nodes = expr.i + 1;

    instructions.push_back({Instruction::BACKWARD, cg, expr.i, (unsigned)ndEdfss.size()});
    ndEdfss.push_back({});
    bwd_kernelss.push_back({});

    auto& ndEdfs = ndEdfss.back();
    ndEdfs.resize(num_nodes);
    auto& bwd_kernels = bwd_kernelss.back();
    auto& nfxs = nfxss.back();
    
    DAO_ASSERT(ndEdfss.size() == nfxss.size(), "ndEdfss.size() != nfxss.size()");
    DAO_COND_WARNING(num_nodes != cg->nodes.size(), "num_nodes != cg->nodes.size()");
    DAO_ASSERT(cg->nodes[expr.i]->dim.size() == 1, "cg->nodes[expr.i]->dim.size() != 1");
    DAO_ASSERT(num_nodes <= cg->nodes.size(), "num_nodes is larger than cg->nodes.size()");
    
    std::vector<bool> needs_derivative(num_nodes, false);
    for (auto i : cg->parameter_nodes){
        auto p_node = static_cast<const dynet::ParameterNodeBase*>(cg->nodes[i]);
        if (p_node->type == dynet::ParameterNodeBase::PARAM) {
            auto param_node = static_cast<const dynet::ParameterNode*>(p_node); 
            if (param_node->params.p) {
                needs_derivative[i] = param_node->params.p->is_updated();
            }
            else {
                needs_derivative[i] = param_node->lparams.p->is_updated();
            }
        }
        else if (p_node->type == dynet::ParameterNodeBase::LOOKUP) {
            auto lookup_node = static_cast<const dynet::LookupNode*>(p_node);
            needs_derivative[i] = lookup_node->params.p->is_updated();
        }
        else {DAO_ERROR("Unknown ParameterNodeBase type");}
    }

    for (unsigned ni = 0; ni < num_nodes; ++ni) {
      bool nd = needs_derivative[ni];
      for (auto arg : cg->nodes[ni]->args)
        nd |= needs_derivative[arg];
      needs_derivative[ni] = nd;
    }

    std::vector<bool> in_computation(num_nodes, false);
    std::vector<bool> first_visited(num_nodes, true);
    in_computation.back() = true;
    for (int i = 0; i < num_nodes; ++i) {
        const auto node_fx = nfxs[i];  // f(x_1, x_2, ..., x_arity), which
                                      // was previously computed by forward.
        DAO_ASSERT(node_fx, "node_fx is nullptr %d, %s", i, cg->nodes[i]->as_dummy_string().c_str());
        auto node_dEdfx = ndEdfs[i] = new dynet::Tensor();  // dE/df(x_1, x_2, ..., x_arity)
        node_dEdfx->name = cg->nodes[i]->as_dummy_string() + "_dEdfx";
        node_dEdfx->d = node_fx->d;
        node_dEdfx->device = node_fx->device;
        node_dEdfx->mem_pool = dynet::DeviceMempool::DEDFS;
    }
    for (int i = num_nodes - 1; i >= 0; --i) {
        if (!in_computation[i] || !needs_derivative[i]) continue; 
        DAO_INFO_LEVEL(1, "Symbolic BWD %d", i);
        const Node* node = cg->nodes[i];
        for (auto arg : node->args)
            in_computation[arg] = true;
        if (node->backward_inplaced()) continue;
        // std::vector<const Tensor*> xs(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg: node->args) {
            if (needs_derivative[arg]) {
                struct BWDKernel bwdKernel;
                bwdKernel.ai = ai;
                bwdKernel.fx_idx = i;

                std::vector<const dynet::Tensor*> xs;
                for (auto x: node->args) {
                    xs.push_back(nfxs[x]);
                    if (cg->nodes[x]->forward_inplaced()) {
                        xs.back() = nfxs[cg->nodes[x]->args[0]];
                    }
                }
                DAO::Kernel kernel;
                kernel.set_inputs(std::move(xs));
                
                if (node->forward_inplaced()) kernel.set_inputs(nfxs[node->args[0]]); 
                else kernel.set_inputs(nfxs[i]);

                auto dEdfxai = cg->nodes[arg]->backward_inplaced()? cg->nodes[arg]->args[0] : arg;

                if (first_visited[dEdfxai]) {
                    kernel.set_zeroed(ndEdfs[dEdfxai]);
                    first_visited[dEdfxai] = false;
                }
                else kernel.set_outputs(ndEdfs[dEdfxai]);

                if (node->aux_storage_size()) {
                    kernel.set_inputs(fwd_kernelss.back()[i].tmp_tensor);
                }
                
                if (i != num_nodes - 1) {  // for all but the last node
                    DAO_ASSERT(!first_visited[i], "first_visited[%u, %s] = true", i, cg->nodes[i]->as_dummy_string().c_str());
                    kernel.set_inputs(ndEdfs[i]);
                }
                allocator.Register(std::move(kernel));
                bwd_kernels.push_back(std::move(bwdKernel));
            }
            ai ++;
        }
    }

    acc_kernelss.push_back({});
    auto& acc_kernels = acc_kernelss.back();

    for (VariableIndex i : cg->parameter_nodes) {
        if (!needs_derivative[i]) continue;
        dynet::ParameterNodeBase* bnode = static_cast<dynet::ParameterNodeBase*>(cg->nodes[i]);
        dynet::Tensor* grads = nullptr;
        if (bnode->type == dynet::ParameterNodeBase::PARAM) {
            assert(!bnode->aux_storage_size());
            dynet::ParameterNode* pnode = static_cast<dynet::ParameterNode*>(bnode);
            if (pnode->params.p) {
                updated_params.insert(pnode->params.p.get());
                grads = &pnode->params.get_storage().g;
            }
            else {
                // kernel.set_inputs(ndEdfs[i]).set_outputs(&pnode->lparams.get_storage().all_grads);
                updated_params.insert(pnode->lparams.p.get());
                grads = &pnode->lparams.get_storage().all_grads;
            }
        }
        else if (bnode->type == dynet::ParameterNodeBase::LOOKUP) {
            dynet::LookupNode* lnode = static_cast<dynet::LookupNode*>(bnode);
            updated_params.insert(lnode->params.p.get());
            grads = &lnode->params.get_storage().all_grads;
        }
        else {DAO_ERROR("Unknown ParameterNodeBase type");}
        DAO::Kernel kernel;
        kernel.set_inputs(ndEdfs[i]).set_outputs(grads);
        auto& tmp = fwd_kernelss.back()[i].tmp_tensor;
        if (tmp) kernel.set_inputs(tmp);
        allocator.Register(std::move(kernel));
        acc_kernels.push_back({bnode, ndEdfs[i], grads, tmp});
    }

    DAO_ASSERT(sanity_check(), "sanity_check() failed");
    timer.stop("symbolic_backward");
}

void Engine::run(
    size_t* max_gpu_usage,
    size_t* max_cpu_usage 
) {
    timer.start("run");
    for (auto o: outputs) {
        allocator.free(o);
        delete o;
    }
    outputs.clear();
    allocator.finish_register();
    for (auto& inst: instructions) {
        if (inst.opcode == Instruction::FORWARD) {
            DAO_INFO("DAO Exec Forward");
            timer.start("run forward");
            run_forward(inst); 
            timer.stop("run forward");
        } else if (inst.opcode == Instruction::BACKWARD) {
            DAO_INFO("DAO Exec Backward");
            timer.start("run backward");
            run_backward(inst);
            timer.stop("run backward");
        } else if (inst.opcode == Instruction::UPDATE) {
            DAO_INFO("DAO Exec Update");
            timer.start("run update");
            run_update(inst); 
            timer.stop("run update");
        }
    }
    if (max_gpu_usage) *max_gpu_usage = std::max(*max_gpu_usage, allocator.max_gpu_usage);
    if (max_cpu_usage) *max_cpu_usage = std::max(*max_cpu_usage, allocator.max_cpu_usage);
    evaluated = true;
    reset();
    timer.stop("run");
}

void Engine::reset() {
    for (auto& nfxs: nfxss) {
        for (size_t i = 0; i < nfxs.size() - 1; ++i) {
            delete nfxs[i];
        }
        if (!outputs.count(nfxs.back())) delete nfxs.back();
    }
    nfxss.clear();

    for (auto& ndEdfs: ndEdfss) {
        for (auto& ndEdf: ndEdfs) {
            delete ndEdf;
        }
    }
    ndEdfss.clear();

    for (auto& fwdtmps: fwdtmpss) {
        for (auto& fwdtmp: fwdtmps) {
            delete fwdtmp;
        }
    }

    fwdtmpss.clear();
    instructions.clear();
    fwd_kernelss.clear();
    bwd_kernelss.clear();
    upd_kernelss.clear();
    acc_kernelss.clear();
    n_fwd_inplaced = 0;
    evaluated = false;
    updated_params.clear();

    allocator.reset();
}

void Engine::report(std::ostream& o) const {
    o << "-------------Engine----------------\n";
    allocator.display(o);
    timer.show(o);
    o << "----------Engine END---------------\n";
} 

void Engine::run_forward(Instruction& inst) {
    auto& cg = inst.cg;
    auto& nfxs = nfxss[inst.idx];
    auto& fwdtmps = fwdtmpss[inst.idx];
    auto& fwd_kernels = fwd_kernelss[inst.idx];

    assert(fwd_kernels.size() == cg->nodes.size());

    for (dynet::VariableIndex i = 0; i < fwd_kernels.size(); ++i) {
        auto node = cg->nodes[i];
        if (node->forward_inplaced()) continue;
        auto& node_fx = nfxs[i];
        auto& fwd_kernel = fwd_kernels[i];
        std::vector<const Tensor*> xs;
        for (auto x: node->args) {
            xs.push_back(nfxs[x]);
        }
        
        if (DAO::offload_profiling) {
            if (const char* name = node2str(node)) {
                timer.start("FWD " + std::string(name));
            }
            else timer.start("FWD " + node->as_dummy_string());
        }

        allocator.prepare();


        if (verbose) {
            std::vector<std::string> input_sizes;
            for (auto x: node->args) {
                std::stringstream ss;
                ss << cg->nodes[x]->as_dummy_string() << ":" <<  nfxs[x]->d;
                input_sizes.push_back(ss.str());
            }
            std::stringstream ss;
            ss << node_fx->d;
            DAO_INFO_LEVEL(1, "FWD %s = %s", ss.str().c_str(), node->as_string(input_sizes).c_str());
        }
        
        if (debug_mode) {
            
            for (auto arg: node->args) {
                DAO_ASSERT(allocator.check_on_gpu(nfxs[cg->nodes[arg]->forward_inplaced()?cg->nodes[arg]->args[0]:arg]), "x->v is nullptr");
            }
            DAO_ASSERT(allocator.check_on_gpu(node_fx), "node_fx->v is nullptr");
            DAO_ASSERT(!node->aux_storage_size() || (fwd_kernel.tmp_tensor && allocator.check_on_gpu(fwd_kernel.tmp_tensor)), "!node->aux_storage_size() || fwd_kernel.tmp_tensor");
            DAO_ASSERT(node_fx->v, "node_fx->v is nullptr");
        }

        if (fwd_kernel.tmp_tensor) {
            node->aux_mem = fwd_kernel.tmp_tensor->v;
            DAO_INFO("N%u, tmp-P %p, aux_storage_size = %lu, size=%u, v %p", i, fwd_kernel.tmp_tensor, node->aux_storage_size(), fwd_kernel.tmp_tensor->d.size(), fwd_kernel.tmp_tensor->v);
        }

        for(auto idx: node->args) {
            auto& xnode = cg->nodes[idx];
            if (xnode->forward_inplaced()) {
                nfxs[idx]->v = nfxs[xnode->args[0]]->v;
            }
        }

        node->forward(xs, *node_fx);
        allocator.complete();

        if (DAO::offload_profiling) {
            cudaDeviceSynchronize();
            timer.stop();
        }
    }

    outputs.insert(nfxs.back());
}

void initialize() {
    dao_allocator.init((size_t)(DAO::cpu_mem * (1 << 20)), (size_t)(DAO::gpu_mem * (1 << 20)), 0, 0);
    dao_allocator.set_compute_stream(DAO::default_stream);
}

void Engine::run_backward(Instruction& inst) {
    DAO_ASSERT(inst.opcode == Instruction::BACKWARD, "inst.opcode != Instruction::BACKWARD");
    auto& cg = inst.cg;
    auto& nfxs = nfxss[inst.idx];
    auto& ndEdfs = ndEdfss[inst.idx];
    auto& bwd_kernels = bwd_kernelss[inst.idx];
    auto& acc_kernels = acc_kernelss[inst.idx];

    ndEdfs.back()->v = cg->nodes.back()->device->kSCALAR_ONE;

    for (auto& bwd_kernel: bwd_kernels) {
        auto node = cg->nodes[bwd_kernel.fx_idx];
        
        if (DAO::offload_profiling){
            if (const char* name = node2str(node)) {
                timer.start("FWD " + std::string(name));
            }
            else {
                timer.start("FWD " + node->as_dummy_string());
            }
        }
        
        if (DAO::offload_profiling){
            if (const char* name = node2str(node)) {
                timer.start("FWD " + std::string(name) + " prepare");
            }
            else {
                timer.start("FWD " + node->as_dummy_string() + " prepare");
            }
        }
        allocator.prepare();
        if (DAO::offload_profiling)
        {
            cudaDeviceSynchronize();
            timer.stop();
        }

        if (DAO::offload_profiling){
            if (const char* name = node2str(node)) {
                timer.start("FWD " + std::string(name) + " exec");
            }
            else {
                timer.start("FWD " + node->as_dummy_string() + " exec");
            }
        }
        std::vector<const Tensor*> xs;
        for (auto arg: node->args) {
            auto& nfx = nfxs[arg];
            auto& xnode = cg->nodes[arg];
            if (xnode->forward_inplaced()) {
                nfx->v = nfxs[xnode->args[0]]->v;
            }
            xs.push_back(nfx);
        }
        auto& node_fx = nfxs[bwd_kernel.fx_idx];
        auto& node_dEdfx = ndEdfs[bwd_kernel.fx_idx];
        auto& arg = node->args[bwd_kernel.ai];
        auto& node_dEdxai = ndEdfs[arg];
        auto& ainode = cg->nodes[arg];

        if (node->forward_inplaced()) node_fx->v = nfxs[node->args[0]]->v;
        if (ainode->backward_inplaced()) node_dEdxai->v = ndEdfs[ainode->args[0]]->v;
        if (node->aux_storage_size()) node->aux_mem = fwd_kernelss[inst.idx][bwd_kernel.fx_idx].tmp_tensor->v;

        if (debug_mode) {
            assert(!node->backward_inplaced());
            for (auto arg: node->args) {
                auto xnode = cg->nodes[arg];
                DAO_ASSERT(nfxs[arg]->v, "nfxs[xnode]->v is nullptr");
                DAO_ASSERT(allocator.check_on_gpu(nfxs[xnode->forward_inplaced()?xnode->args[0]:arg]), "x->v is nullptr");
            }
            DAO_ASSERT(allocator.check_on_gpu(node->forward_inplaced()?nfxs[node->args[0]]:node_fx), "node_fx.v is nullptr");
            DAO_ASSERT(bwd_kernel.fx_idx == ndEdfs.size() - 1 || allocator.check_on_gpu(node_dEdfx), "node_dEdfx.v is nullptr");
            DAO_ASSERT(allocator.check_on_gpu(ainode->backward_inplaced()?ndEdfs[ainode->args[0]]:node_dEdxai), "node_dEdxai.v is nullptr");
            DAO_ASSERT(node_dEdfx->v && node_dEdxai->v, "node_dEdfx->v && node_dEdxai->v");
            DAO_INFO_LEVEL(1, "DAO BWD %u %p = %s", bwd_kernel.ai, node_dEdxai->v, node->as_dummy_string().c_str());
        }

        node->backward(xs, *node_fx, *node_dEdfx, bwd_kernel.ai, *node_dEdxai);

        if (DAO::offload_profiling)
        {
            cudaDeviceSynchronize();
            timer.stop();
        }

        if (DAO::offload_profiling){
            if (const char* name = node2str(node)) {
                timer.start("FWD " + std::string(name) + " exec");
            }
            else {
                timer.start("FWD " + node->as_dummy_string() + " exec");
            }
        }

        allocator.complete();

        if (DAO::offload_profiling)
        {
            cudaDeviceSynchronize();
            timer.stop();
        }

        if (DAO::offload_profiling)
        {
            cudaDeviceSynchronize();
            timer.stop();
        }
    }

    DAO_INFO_LEVEL(1, "BWD finished");

    if (DAO::offload_profiling) timer.start("accumulate_grad");
    for (auto& acc_grad_kernel: acc_kernels) {
        auto& node = acc_grad_kernel.pnode;
        allocator.prepare();
        DAO_ASSERT(!(node->aux_storage_size()) || acc_grad_kernel.tmp, "!node->aux_storage_size()=%lu || acc_grad_kernel.tmp = %p", node->aux_storage_size(), acc_grad_kernel.tmp);
        DAO_ASSERT(allocator.check_on_gpu(acc_grad_kernel.ndEdf), "acc_grad_kernel.ndEdf.v is nullptr");
        DAO_ASSERT(allocator.check_on_gpu(acc_grad_kernel.grads), "acc_grad_kernel.grads.v is nullptr");
        DAO_ASSERT(!acc_grad_kernel.tmp || allocator.check_on_gpu(acc_grad_kernel.tmp), "tmp.v is nullptr");
        if (acc_grad_kernel.tmp)
            node->aux_mem = acc_grad_kernel.tmp->v;
        acc_grad_kernel.pnode->accumulate_grad(*acc_grad_kernel.ndEdf);
        allocator.complete();
    }
    if (DAO::offload_profiling) {
        cudaDeviceSynchronize();
        timer.stop("accumulate_grad");}
}

void Engine::symbolic_update() {
    timer.start("symbolic_update");
    instructions.push_back({Instruction::UPDATE, nullptr, 0, (unsigned)upd_kernelss.size()});
    upd_kernelss.push_back({});
    auto& upd_kernels = upd_kernelss.back();
    bool old_enabled = enabled;
    enabled = true;
    const auto& params = trainer->model->updated_parameters_list();
    const auto& lparams = trainer->model->updated_lookup_parameters_list();
    // const float gscale = trainer->clip_gradients();
    // DAO_INFO_LEVEL(0, "gscale = %f", gscale);
    /** TODO: add weight decay*/ 
    const float gscale = 1.0f;
    for (size_t i = 0; i < params.size(); ++i) {
        auto& param = params[i];
        assert(param->updated);
        if (updated_params.count(param.get())) {
            trainer->update_params(gscale, i);
            UPDKernel upd_kernel;
            upd_kernel.gscale = gscale;
            upd_kernel.values = allocator.get_last_kernel()._inputs;
            upd_kernel.p = param;
            upd_kernel.lp = nullptr;
            upd_kernels.push_back(std::move(upd_kernel));            
        }
        else {
            if (debug_mode)
                DAO_WARNING("%s is not updated", param->name.c_str());
        }
    }
    for (size_t i = 0; i < lparams.size(); ++i) {
        auto& lparam = lparams[i];
        assert(lparam->updated);
        if (updated_params.count(lparam.get())) {
            trainer->update_lookup_params(gscale, i);
            UPDKernel upd_kernel;
            upd_kernel.gscale = gscale;
            upd_kernel.values = allocator.get_last_kernel()._inputs;
            upd_kernel.lp = lparam;
            upd_kernel.p = nullptr;
            upd_kernels.push_back(std::move(upd_kernel));   
        }
        else {
            if (debug_mode)
                DAO_WARNING("%s is not updated", lparam->name.c_str());
        }
    }

    ++trainer->updates;
    ++trainer->updates_since_status; 

    enabled = old_enabled;
    // clear the updated params
    updated_params.clear();
    DAO_ASSERT(sanity_check(), "sanity_check() failed");
    timer.stop("symbolic_update");
}

void Engine::run_update(Instruction& inst) {
    DAO_ASSERT(inst.opcode == Instruction::UPDATE, "inst.opcode != Instruction::UPDATE");
    auto& upd_kernels = upd_kernelss[inst.idx];
    for (auto& upd_kernel: upd_kernels) {
        allocator.prepare();
        for (auto& value: upd_kernel.values) {
            DAO_ASSERT(allocator.check_on_gpu(value), "value.v is nullptr");
        }
        trainer->update_rule(upd_kernel.gscale, upd_kernel.values);
        if (upd_kernel.p) {
            upd_kernel.p->clear();
        } else {
            upd_kernel.lp->clear();
        }
        allocator.complete();
    }
}

Engine::~Engine() {
    for (auto o: outputs) {
        allocator.free(o);
        delete o;
    }
    reset();
}

void Engine::dump_statistics(const std::string& filename) const {
    allocator.dump_memory_breakdown(filename);
    timer.save(filename);
}

} // namespace DAO