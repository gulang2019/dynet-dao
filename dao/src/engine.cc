#include "DAO/engine.h" 

#include <dynet/devices.h>
#include <dynet/param-nodes.h>
#include <dynet/globals.h>
#include <string>

namespace DAO { 

using dynet::Node;
using dynet::Tensor; 
using dynet::VariableIndex;

Engine::Engine(): allocator(dao_allocator) {}

std::vector<float> Engine::as_vector(const dynet::Tensor& tensor) {
    return std::move(allocator.get_values((TensorUID)&tensor));
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
    nfxs.resize(expr.i + 1);

    dynet::VariableIndex param_idx = 0;
    for (dynet::VariableIndex num_nodes_evaluated = 0; 
        num_nodes_evaluated <= expr.i; ++num_nodes_evaluated) {
        const Node* node = cg->nodes[num_nodes_evaluated];
        if (node->forward_inplaced()) continue;
        std::vector<dynet::VariableIndex> xs_idx;
        for (VariableIndex arg : node->args) {
            xs_idx.push_back(arg);
            if (cg->nodes[arg]->forward_inplaced()) {
                xs_idx.back() = cg->nodes[arg]->args[0];
            }
        }
        auto& node_fx = nfxs[num_nodes_evaluated] = new dynet::Tensor();
        node_fx->d = node->dim;
        // Get the device
        DYNET_ASSERT(node->device != nullptr,
        "Attempt to access null device in "
        "SimpleExecutionEngine::incremental_forward");
        node_fx->device = node->device;
        node_fx->mem_pool = dynet::DeviceMempool::FXS;
        // Get the memory to store f(xs)

        FWDKernel fwd_kernel;
        DAO::Kernel kernel;
        fwd_kernel.fx_idx = num_nodes_evaluated;
        fwd_kernel.xs_idx = std::move(xs_idx);
        fwd_kernel.tmp_idx = (dynet::VariableIndex)(-1);
        if (node->aux_mem) {
            fwdtmps.push_back(new dynet::Tensor());
            fwdtmps.back()->d = dynet::Dim({(unsigned)(node->aux_storage_size() / sizeof(float) + 1)});
            fwd_kernel.tmp_idx = fwdtmps.size() - 1;
            kernel.set_inputs(fwdtmps.back());
        }
        std::vector<const dynet::Tensor*> xs;
        for (auto x: fwd_kernel.xs_idx) {
            xs.push_back(nfxs[x]);
        }
        kernel.set_inputs(xs).set_outputs(node_fx);
        if (param_idx < cg->parameter_nodes.size()
        && cg->parameter_nodes[param_idx] == num_nodes_evaluated) {
            auto param_node = static_cast<const dynet::ParameterNode*>(node);
            if(param_node->params.p) // Parameter
                kernel.set_inputs(&param_node->params.get_storage().values);
            else kernel.set_inputs(&param_node->lparams.get_storage().all_values);
            param_idx++;
        }
        allocator.Register(std::move(kernel));
        fwd_kernels.push_back(std::move(fwd_kernel));
    }
    allocator.set_global(nfxs.back());
    timer.stop("symbolic_forward");
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
    for (auto i : cg->parameter_nodes)
        needs_derivative[i] = true;

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
        auto node_dEdfx = ndEdfs[i] = new dynet::Tensor();  // dE/df(x_1, x_2, ..., x_arity)
        node_dEdfx->d = node_fx->d;
        node_dEdfx->device = node_fx->device;
        node_dEdfx->mem_pool = dynet::DeviceMempool::DEDFS;
    }
    for (int i = num_nodes - 1; i >= 0; --i) {
        if (!in_computation[i] || !needs_derivative[i]) continue; 
        DAO_INFO("Symbolic BWD %d", i);
        const Node* node = cg->nodes[i];
        for (auto arg : node->args)
            in_computation[arg] = true;
        if (node->backward_inplaced()) continue;
        std::vector<dynet::VariableIndex> xs_idx(node->arity());
        // std::vector<const Tensor*> xs(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg: node->args) {
            xs_idx[ai] = arg;
            if (cg->nodes[arg]->forward_inplaced()) {
                xs_idx[ai] = cg->nodes[arg]->args[0];
            }
            ++ai;
        }
        ai = 0;
        for (VariableIndex arg: node->args) {
            if (needs_derivative[arg]) {
                struct BWDKernel bwdKernel;
                bwdKernel.ai = ai;
                bwdKernel.xs_idx = xs_idx;
                bwdKernel.fx_idx = i;
                bwdKernel.dEdfx_idx = i;
                bwdKernel.dEdxai_idx = arg;
                if (cg->nodes[arg]->backward_inplaced())
                    bwdKernel.dEdxai_idx = cg->nodes[arg]->args[0];
                std::vector<const dynet::Tensor*> xs;
                for (auto x: xs_idx) {
                    xs.push_back(nfxs[x]);
                }
                DAO::Kernel kernel;
                kernel.set_inputs(std::move(xs))
                    .set_inputs(nfxs[bwdKernel.fx_idx]);
                if (first_visited[bwdKernel.dEdxai_idx]) {
                    kernel.set_zeroed(ndEdfs[bwdKernel.dEdxai_idx]);
                    first_visited[bwdKernel.dEdxai_idx] = false;
                }
                else kernel.set_outputs(ndEdfs[bwdKernel.dEdxai_idx]);
                if (i != num_nodes - 1) {  // for all but the last node
                    DAO_ASSERT(!first_visited[bwdKernel.dEdfx_idx], "first_visited[%u] = true", bwdKernel.dEdfx_idx);
                    kernel.set_inputs(ndEdfs[bwdKernel.dEdfx_idx]);
                }
                allocator.Register(std::move(kernel));
                bwd_kernels.push_back(std::move(bwdKernel));
            }
            ai ++;
        }
    }

    for (VariableIndex i : cg->parameter_nodes) {
        dynet::ParameterNode* pnode = static_cast<dynet::ParameterNode*>(cg->nodes[i]);
        DAO::Kernel kernel;
        if (pnode->params.p) 
            kernel.set_inputs(ndEdfs[i]).set_outputs(&pnode->params.get_storage().g);
        else 
            kernel.set_inputs(ndEdfs[i]).set_outputs(&pnode->lparams.get_storage().all_grads);
        allocator.Register(std::move(kernel));
    }
    timer.stop("symbolic_backward");
}

void Engine::run() {
    allocator.finish_register();
    for (auto o: outputs) {
        allocator.free(o);
    }
    outputs.clear();
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
            run_update(inst); 
        }
    }

    for (auto& nfxs: nfxss) {
        outputs.insert(nfxs.back());
        for (auto& nfx: nfxs) {
            if (outputs.count(nfx) == 0)
                delete nfx;
        }
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
    allocator.free_intermidiates();
}

void Engine::report(std::ostream& o) {
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

    for (auto& fwd_kernel: fwd_kernels) {
        auto& node_fx = nfxs[fwd_kernel.fx_idx];
        std::vector<const Tensor*> xs;
        auto node = cg->nodes[fwd_kernel.fx_idx];
        for (auto x: fwd_kernel.xs_idx) {
            xs.push_back(nfxs[x]);
        }

        std::vector<std::string> input_sizes;
        for (auto x: xs) {
            std::stringstream ss;
            ss << x->d;
            input_sizes.push_back(ss.str());
        }
        std::stringstream ss;
        ss << node_fx->d;
        DAO_INFO_LEVEL(1, "FWD %s = %s", ss.str().c_str(), node->as_string(input_sizes).c_str());
        allocator.prepare();
        for (auto x: xs) {
            DAO_ASSERT(allocator.check_on_gpu(x), "x->v is nullptr");
        }
        DAO_ASSERT(allocator.check_on_gpu(node_fx), "node_fx->v is nullptr");

        if (node->aux_mem) {
            DAO_ASSERT(fwd_kernel.tmp_idx != (dynet::VariableIndex)(-1), "fwd_kernel.tmp_idx == -1");
            auto tmp_tensor = fwdtmps[fwd_kernel.tmp_idx];
            DAO_ASSERT(allocator.check_on_gpu(tmp_tensor), "node->aux_mem is nullptr");
            node->aux_mem = tmp_tensor->v;
        } 
        DAO_ASSERT(node_fx->v, "node_fx->v is nullptr");

        node->forward(xs, *node_fx);
        allocator.complete();
    }
}

void initialize(int argc, char** argv) {
    dao_allocator.init((size_t)(DAO::cpu_mem * (1 << 20)), (size_t)(DAO::gpu_mem * (1 << 20)), 0, 0);
    dao_allocator.set_compute_stream(DAO::default_stream);
}

void Engine::run_backward(Instruction& inst) {
    DAO_ASSERT(inst.opcode == Instruction::BACKWARD, "inst.opcode != Instruction::BACKWARD");
    auto& cg = inst.cg;
    auto& nfxs = nfxss[inst.idx];
    auto& ndEdfs = ndEdfss[inst.idx];
    auto& bwd_kernels = bwd_kernelss[inst.idx];

    ndEdfs.back()->v = cg->nodes.back()->device->kSCALAR_ONE;

    for (auto& bwd_kernel: bwd_kernels) {
        allocator.prepare();
        std::vector<const Tensor*> xs;
        for (auto x: bwd_kernel.xs_idx) {
            xs.push_back(nfxs[x]);
            DAO_ASSERT(allocator.check_on_gpu(nfxs[x]), "nfxs[x].v is nullptr");
        }
        auto node = cg->nodes[bwd_kernel.fx_idx];
        auto& node_fx = nfxs[bwd_kernel.fx_idx];
        auto& node_dEdfx = ndEdfs[bwd_kernel.dEdfx_idx];
        auto& node_dEdxai = ndEdfs[bwd_kernel.dEdxai_idx];

        DAO_ASSERT(allocator.check_on_gpu(node_fx), "node_fx.v is nullptr");
        DAO_ASSERT(bwd_kernel.dEdfx_idx == ndEdfs.size() - 1 || allocator.check_on_gpu(node_dEdfx), "node_dEdfx.v is nullptr");
        DAO_ASSERT(allocator.check_on_gpu(node_dEdxai), "node_dEdxai.v is nullptr");

        DAO_INFO("DAO BWD %u %p = %s", bwd_kernel.ai, node_dEdxai->v, node->as_dummy_string().c_str());

        node->backward(xs, *node_fx, *node_dEdfx, bwd_kernel.ai, *node_dEdxai);

        allocator.complete();
    }

    for (VariableIndex i : cg->parameter_nodes) {
        dynet::ParameterNodeBase* pnode = static_cast<dynet::ParameterNodeBase*>(cg->nodes[i]);
        allocator.prepare();
        DAO_ASSERT(allocator.check_on_gpu(ndEdfs[i]), "ndEdfs[i].v is nullptr");
        pnode->accumulate_grad(*ndEdfs[i]);
        allocator.complete();
    }
}

void Engine::run_update(Instruction&) {}
} // namespace DAO