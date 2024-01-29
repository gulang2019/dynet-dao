#include "DAO/engine.h" 

#include <dynet/devices.h>
#include <dynet/param-nodes.h>
#include <dynet/globals.h>
#include <dynet/weight-decay.h>


#include <string>

namespace DAO { 

using dynet::Node;
using dynet::Tensor; 
using dynet::VariableIndex;

bool enabled = false;

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
    return n_kernels == (allocator.all_accesses.size() + n_fwd_inplaced);
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

        // for inplaced node, we only need to copy the pointer
        if (node->forward_inplaced()) {
            n_fwd_inplaced ++;
            continue;
        }

        auto& fwd_kernel = fwd_kernels[num_nodes_evaluated];
        DAO::Kernel kernel;
        fwd_kernel.tmp_tensor = nullptr;
        if (node->aux_storage_size()) {
            fwd_kernel.tmp_tensor = new dynet::Tensor();
            assert(fwd_kernel.tmp_tensor);
            fwd_kernel.tmp_tensor->d = dynet::Dim({(unsigned)(node->aux_storage_size() / sizeof(float) + 1)});
            fwdtmps.push_back(fwd_kernel.tmp_tensor);
            kernel.set_inputs(fwd_kernel.tmp_tensor);
        }
        std::vector<const dynet::Tensor*> xs;
        for (auto idx: node->args) {
            xs.push_back(nfxs[idx]);
            // if it is inplaced, we prepare the child node for the uniqueness; 
            if (cg->nodes[idx]->forward_inplaced()) {
                xs.back() = nfxs[cg->nodes[idx]->args[0]];
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
    allocator.set_global(nfxs.back());
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
        DAO_ASSERT(node_fx, "node_fx is nullptr %d, %s", i, cg->nodes[i]->as_dummy_string().c_str());
        auto node_dEdfx = ndEdfs[i] = new dynet::Tensor();  // dE/df(x_1, x_2, ..., x_arity)
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

void Engine::run() {
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
            run_update(inst); 
        }
    }
    evaluated = true;
    reset();
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
        
        allocator.prepare();

        if (debug_mode) {
            std::vector<std::string> input_sizes;
            for (auto x: node->args) {
                std::stringstream ss;
                ss << cg->nodes[x]->as_dummy_string() << ":" <<  nfxs[x]->d;
                input_sizes.push_back(ss.str());
            }
            std::stringstream ss;
            ss << node_fx->d;
            DAO_INFO_LEVEL(1, "FWD %s = %s", ss.str().c_str(), node->as_string(input_sizes).c_str());

            for (auto arg: node->args) {
                DAO_ASSERT(allocator.check_on_gpu(nfxs[cg->nodes[arg]->forward_inplaced()?cg->nodes[arg]->args[0]:arg]), "x->v is nullptr");
            }
            DAO_ASSERT(allocator.check_on_gpu(node_fx), "node_fx->v is nullptr");
            DAO_ASSERT(!node->aux_storage_size() || (fwd_kernel.tmp_tensor && allocator.check_on_gpu(fwd_kernel.tmp_tensor)), "!node->aux_storage_size() || fwd_kernel.tmp_tensor");
            DAO_ASSERT(node_fx->v, "node_fx->v is nullptr");
        }

        if (fwd_kernel.tmp_tensor) {
            node->aux_mem = fwd_kernel.tmp_tensor->v;
        }

        for(auto idx: node->args) {
            auto& xnode = cg->nodes[idx];
            if (xnode->forward_inplaced()) {
                nfxs[idx]->v = nfxs[xnode->args[0]]->v;
            }
        }

        node->forward(xs, *node_fx);
        allocator.complete();
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
        allocator.prepare();
        auto node = cg->nodes[bwd_kernel.fx_idx];
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
        if (ainode->forward_inplaced()) node_dEdxai->v = ndEdfs[ainode->args[0]]->v;
        if (node->aux_storage_size()) node->aux_mem = fwd_kernelss[inst.idx][bwd_kernel.fx_idx].tmp_tensor->v;

        if (debug_mode) {
            for (auto arg: node->args) {
                auto xnode = cg->nodes[arg];
                DAO_ASSERT(nfxs[arg]->v, "nfxs[xnode]->v is nullptr");
                DAO_ASSERT(allocator.check_on_gpu(nfxs[xnode->forward_inplaced()?xnode->args[0]:arg]), "x->v is nullptr");
            }
            DAO_ASSERT(allocator.check_on_gpu(node->forward_inplaced()?nfxs[node->args[0]]:node_fx), "node_fx.v is nullptr");
            DAO_ASSERT(bwd_kernel.fx_idx == ndEdfs.size() - 1 || allocator.check_on_gpu(node_dEdfx), "node_dEdfx.v is nullptr");
            DAO_ASSERT(allocator.check_on_gpu(node->backward_inplaced()?ndEdfs[cg->nodes[arg]->args[0]]:node_dEdxai), "node_dEdxai.v is nullptr");
            DAO_ASSERT(node_dEdfx->v && node_dEdxai->v, "node_dEdfx->v && node_dEdxai->v");
            DAO_INFO_LEVEL(1, "DAO BWD %u %p = %s", bwd_kernel.ai, node_dEdxai->v, node->as_dummy_string().c_str());
        }

        node->backward(xs, *node_fx, *node_dEdfx, bwd_kernel.ai, *node_dEdxai);

        allocator.complete();
    }

    DAO_INFO_LEVEL(1, "BWD finished");


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
}

void Engine::symbolic_update() {
    instructions.push_back({Instruction::UPDATE, nullptr, 0, (unsigned)upd_kernelss.size()});
    upd_kernelss.push_back({});
    auto& upd_kernels = upd_kernelss.back();
    bool old_enabled = enabled;
    enabled = true;
    const auto& params = trainer->model->parameters_list();
    const auto& lparams = trainer->model->lookup_parameters_list();
    const float gscale = trainer->clip_gradients();
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
                DAO_WARNING("param[%u] is not updated", i);
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
            DAO_WARNING("lparam[%u] is not updated", i);
        }
    }

    ++trainer->updates;
    ++trainer->updates_since_status; 

    enabled = old_enabled;
    // clear the updated params
    updated_params.clear();
    DAO_ASSERT(sanity_check(), "sanity_check() failed");
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

} // namespace DAO