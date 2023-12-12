#include <string>
#include <vector>

#include "DAO/interface-dynet.h"
#include "DAO/globals.h"
#include "DAO/generator.h"

#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include "dynet/timing.h"
#include "dynet/param-nodes.h"
#include "dynet/devices.h"


using std::vector;
using std::string;
using dynet::Node;
using dynet::DeviceMempool;


namespace DAO {


void AsyncExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  backward_computed = 0;
}

void AsyncExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& AsyncExecutionEngine::forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& AsyncExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& AsyncExecutionEngine::get_value(VariableIndex i) {
    DAO_ERROR("get_value not supported");
    Tensor dummy; 
  return dummy;
}

const Tensor& AsyncExecutionEngine::get_gradient(VariableIndex i) {
  DAO_ERROR("get_gradient not supported");
  Tensor dummy; 
  return dummy;
}

const Tensor& AsyncExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& AsyncExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(),
    "Out-of-bounds variable access in "
    "AsyncExecutionEngine::incremental_forward()");
  DAO_ASSERT(!cg.nodes[i]->forward_inplaced(), "Inplaced nodes cannot be accessed in AsyncExecutionEngine::incremental_forward()"); 

  if (i >= num_nodes_evaluated) {
    string current_node_name;  // Optionally used for debugging (reused).
    vector<std::shared_ptr<Tensor> > xs(16);  // Container for arguments to nodes (reused).
    nfxs.resize(i+1, nullptr);
    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
        const Node* node = cg.nodes[num_nodes_evaluated];
        if (node->forward_inplaced()) continue;
        xs.resize(node->arity());
        unsigned ai = 0;
        for (VariableIndex arg : node->args) {
          while(cg.nodes[arg]->forward_inplaced()) 
            arg = cg.nodes[arg]->args[0];
          xs[ai] = nfxs[arg];
          DYNET_ARG_CHECK(xs[ai]->device == node->device ||
              node->supports_multidevice(),
              "Attempt to do tensor forward in different devices (nodes " <<
              arg << " and " << num_nodes_evaluated << ")");
          ++ai;
        }
        std::shared_ptr<Tensor>& node_fx = nfxs[num_nodes_evaluated] = std::make_shared<Tensor>();
        node_fx->name = "FWD " + node->as_dummy_string();
        node_fx->d = node->dim; 
        DYNET_ASSERT(node->device != nullptr,
          "Attempt to access null device in "
          "AsyncExecutionEngine::incremental_forward");
        node_fx->device = node->device;
        node_fx->mem_pool = DeviceMempool::FXS;
        DAO::Kernel kernel;
        kernel.set_name(("FWD " + node->as_dummy_string()).c_str());
        kernel.set_outputs(node_fx.get());
        for(auto& x: xs) kernel.set_inputs(x.get());
        kernel.set_impl([node, node_fx, xs, idx=num_nodes_evaluated](Kernel*){
            auto& node_fx_pools = node_fx->device->pools;
            // node_fx->v = static_cast<float*>(
            // node_fx_pools[(int)DeviceMempool::FXS]->allocate(
            //     node->dim.size() * sizeof(float)));
            DAO_ASSERT(node_fx->v, "node_fx->v is null");
            for (auto& x: xs) 
              DAO_ASSERT(x!= nullptr && x->v, "xs[i]->v is null");
            // if (node_fx->v == nullptr) {
            // DYNET_RUNTIME_ERR("Ran out of memory when executing node " <<
            //                     idx << ", allocating FWD memory.");
            // }
            void* aux_mem = nullptr;
            // Is the node requesting extra memory?
            size_t aux_size = node->aux_storage_size();
            if (aux_size) {
                aux_mem = node_fx_pools[(int)DeviceMempool::FXS]->allocate(aux_size);
            if (aux_mem == nullptr)
                DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node "
                                << idx);
            }
            node->aux_mem = aux_mem;
            vector<const Tensor*> xs_;
            for(auto& x: xs) xs_.push_back(x.get());
            node->forward(xs_, *node_fx);
        });
        DAO::push_kernel(std::move(kernel));
    }
  }
  DAO_ASSERT(i < nfxs.size(), "Out-of-bounds variable access in AsyncExecutionEngine::incremental_forward()");
  return *nfxs[i];
}

void AsyncExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(),
               "Mismatched array sizes in AsyncExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size() - 1), full);
}

void AsyncExecutionEngine::backward(VariableIndex from_where, bool full) {
  if (from_where >= nfxs.size()) { incremental_forward(from_where); }
  if (nfxs[from_where]->d.size() != 1) {
    DYNET_INVALID_ARG(
        "backward() can only be called on scalar nodes, but node "
        << from_where << " has dimension: " << nfxs[from_where]->d);
  }

  const unsigned num_nodes = from_where + 1;
  ndEdfs.resize(num_nodes);
  // initialize dE/dE = 1
  // ndEdfs.back().v = cg.nodes.back()->device->kSCALAR_ONE;

  // here we find constant paths to avoid doing extra work
  // by default, a node is constant unless
  //   1) it is a parameter node
  //   2) it depends on a non-constant node
  // (thus, functions of constants and inputs end up being
  //  false in this computation)
  vector<bool> needs_derivative(num_nodes, full);
  if (!full) {
    for (auto i : cg.parameter_nodes)
      if (i <= from_where)
        needs_derivative[i] = true;

    for (unsigned ni = 0; ni < num_nodes; ++ni) {
      bool nd = needs_derivative[ni];
      for (auto arg : cg.nodes[ni]->args)
        nd |= needs_derivative[arg];
      needs_derivative[ni] = nd;
    }
  }

  // Loop in reverse topological order (nodes stored in topological order),
  // considering only nodes that participate in the computation.
  vector<bool> in_computation(num_nodes, false);
  in_computation[num_nodes - 1] = true;
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (in_computation[i]) 
      for (auto arg: cg.nodes[i]->args)
        in_computation[arg] = true;
  }

  // {
  //   DAO::Kernel kernel;
  //   kernel.set_name("free");
  //   for (int i = 0; i < num_nodes; ++i) {
  //     if (in_computation[i] 
  //       || cg.nodes[i]->forward_inplaced()
  //       || needs_derivative[i]) continue;
  //     if (nfxs[i] != nullptr) {
  //       kernel.set_free(nfxs[i].get());
  //     }
  //   }
  //   kernel.set_impl([](DAO::Kernel*){
  //     // do nothing
  //   });
  //   DAO::push_kernel(std::move(kernel));    
  // }

  ndEdfs.resize(num_nodes, nullptr);
  vector<std::shared_ptr<Tensor> > xs(16, nullptr);  // Container for arguments to nodes (reused).
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    if (node->backward_inplaced()) continue;
    auto node_dEdfx = std::make_shared<Tensor>();
    xs.resize(node->arity());
    assert(xs.size() == node->arity());
    vector<VariableIndex> args; 
    vector<unsigned> ais;
    unsigned ai = 0;
    for (auto arg: node->args) {
      for (auto child_node = cg.nodes[arg]; child_node->forward_inplaced(); ){
        arg = child_node->args[0];
        child_node = cg.nodes[arg];
      }
      if (needs_derivative[arg]) {
        args.push_back(arg);
        ais.push_back(ai);
      }
      xs[ai++] = nfxs[arg];      
    }
    auto& node_ndEdfx = ndEdfs[i];
    if (node_ndEdfx == nullptr) {
      node_ndEdfx = std::make_shared<Tensor>();
      node_ndEdfx->d = nfxs[i]->d;
      node_ndEdfx->device = nfxs[i]->device;
      node_ndEdfx->mem_pool = DeviceMempool::DEDFS;
      node_ndEdfx->v = nullptr;
      if (i == from_where) 
        node_ndEdfx->v = nfxs[i]->device->kSCALAR_ONE;
      node_ndEdfx->name = "BWD " + cg.nodes[i]->as_dummy_string();
    }
    for (unsigned j = 0; j < args.size(); ++j) {
      auto arg = args[j];
      auto ai = ais[j];
      auto& node_dEdxai = ndEdfs[arg];  // where to store dE/dx_{ai}.
      if (node_dEdxai == nullptr) {
        node_dEdxai = std::make_shared<Tensor>();
        node_dEdxai->v = nullptr;
        node_dEdxai->d = nfxs[arg]->d;
        node_dEdxai->device = nfxs[arg]->device;
        node_dEdxai->mem_pool = DeviceMempool::DEDFS;
        node_dEdxai->name = "BWD " + cg.nodes[arg]->as_dummy_string();
      }
      DAO::Kernel kernel;
      kernel.set_name(("BWD " + node->as_dummy_string()).c_str());
      for (auto& x: xs) kernel.set_inputs(x.get());
      kernel.set_inputs(nfxs[i].get(), node_dEdfx.get());
      kernel.set_outputs(node_dEdxai.get());
      if (j == args.size() - 1)
        kernel.set_free(nfxs[i].get(), node_dEdfx.get()); // free the memory at last compute
      kernel.set_impl([
        node,
        xs,
        node_fx = nfxs[i], 
        node_dEdfx = ndEdfs[i], 
        ai, 
        node_dEdxai = ndEdfs[arg]](DAO::Kernel*){
          for (auto & ptr: {node_dEdfx, node_dEdxai, node_fx}) {
            DAO_ASSERT(ptr != nullptr && ptr->v !=nullptr, "Backward nullptr detected");
          }
          for (auto& x: xs) {
            DAO_ASSERT(x != nullptr && x->v != nullptr, "Backward nullptr detected");
          }
          // for (auto& ptr: {node_dEdfx, node_dEdxai})
          // if (ptr->v == nullptr) {
          //   ptr->v = static_cast<float*>(
          //     ptr->device->pools[(int)DeviceMempool::DEDFS]->allocate(
          //         ptr->d.size() * sizeof(float)));
          // }
          vector<const Tensor*> xs_;
          for (auto& x: xs) {
            DAO_ASSERT(x != nullptr, "Out-of-bounds variable access in AsyncExecutionEngine::backward()");
            xs_.push_back(x.get());
          }
          node->backward(xs_, *node_fx, *node_dEdfx, ai, *node_dEdxai);
      });
      DAO::push_kernel(std::move(kernel));
    }
  }

  // Accumulate gradients into parameters.
  for (VariableIndex i : cg.parameter_nodes) {
    if (i <= from_where) {
      auto& node_ndEdf = ndEdfs[i];
      DAO_ASSERT(node_ndEdf != nullptr, "Out-of-bounds variable access in AsyncExecutionEngine::backward()");
      DAO_ASSERT(!cg.nodes[i]->backward_inplaced(), "Inplaced nodes cannot be parameter in AsyncExecutionEngine::backward()");
      DAO::Kernel kernel;
      std::ostringstream s;
      s << "ACCUMULATE GRADIENTS " << cg.nodes[i]->as_dummy_string() << " " << i;
      kernel.set_name(s.str().c_str());
      kernel.set_inputs(node_ndEdf.get());
      kernel.set_free(node_ndEdf.get());
      kernel.set_impl([node = cg.nodes[i], node_ndEdf](DAO::Kernel*){
        dynet::ParameterNodeBase* pnode = static_cast<dynet::ParameterNodeBase*>(node);
        pnode->accumulate_grad(*node_ndEdf);
      });
      DAO::push_kernel(std::move(kernel));
    }
  }
  backward_computed = from_where+1;
}

} // namespace DAO 