#include <string>
#include <vector>

#include "DAO/interface-dynet.h"
#include "DAO/globals.h"
#include "DAO/generator.h"
#include "DAO/allocator.h"

#include "dynet/param-nodes.h"
#include "dynet/globals.h"
#include "dynet/timing.h"
#include "dynet/param-nodes.h"
#include "dynet/devices.h"
#include "dynet/globals.h"


using std::vector;
using std::string;
using dynet::Node;
using dynet::DeviceMempool;
using dynet::Device;
using dynet::timer; 
using dynet::profiling_flag; 
using dynet::ParameterNodeBase; 


namespace DAO {

Allocator* OffloadExecutionEngine::allocator = nullptr;

OffloadExecutionEngine::OffloadExecutionEngine(const dynet::ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0) {
      if (allocator == nullptr) {
        allocator = new Allocator(
          cpu_mem,
          gpu_mem,
          cpu_mem_limit,
          gpu_mem_limit
        );
      }
}

void OffloadExecutionEngine::invalidate() {
  num_nodes_evaluated = 0;
  backward_computed = 0;
}

void OffloadExecutionEngine::invalidate(unsigned i) {
  num_nodes_evaluated = i;
}

const Tensor& OffloadExecutionEngine::forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return forward(node_max_index);
}

const Tensor& OffloadExecutionEngine::forward(VariableIndex i) {
  invalidate();
  return incremental_forward(i);
}

const Tensor& OffloadExecutionEngine::get_value(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(),
      "Out-of-bounds variable access in OffloadExecutionEngine::get_value()");
  if (i >= num_nodes_evaluated) {
    incremental_forward(i);
  }
  return nfxs[i];
}

const Tensor& OffloadExecutionEngine::get_gradient(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(),
      "Out-of-bounds variable access in OffloadExecutionEngine::get_value()");
  if (i >= backward_computed) {
    DYNET_RUNTIME_ERR("Requested gradient for node " << i
                      << ", but backward pass was computed from node "
                      << (backward_computed - 1));
  }
  if(cg.nodes[i]->backward_inplaced()){
    DYNET_RUNTIME_ERR("This operation is an inplaced operation, thus no valid gradient");
  }
  return ndEdfs[i];
}

const Tensor& OffloadExecutionEngine::incremental_forward() {
  const VariableIndex node_max_index = (VariableIndex)(cg.nodes.size() - 1);
  return incremental_forward(node_max_index);
}

const Tensor& OffloadExecutionEngine::incremental_forward(VariableIndex i) {
  DYNET_ASSERT(i < cg.nodes.size(),
    "Out-of-bounds variable access in "
    "OffloadExecutionEngine::incremental_forward()");

  // free any old memory if this is a new CG
  if (num_nodes_evaluated == 0)
    for (Device* dev : device_manager->get_devices())
      dev->pools[(int)DeviceMempool::FXS]->free();

  if (i >= num_nodes_evaluated) {
    nfxs.resize(i + 1);
    string current_node_name;  // Optionally used for debugging (reused).
    vector<const Tensor*> xs(16);  // Container for arguments to nodes (reused).

    for (; num_nodes_evaluated <= i; ++num_nodes_evaluated) {
      const Node* node = cg.nodes[num_nodes_evaluated];
      if (profiling_flag) {
        current_node_name = "FWD " + node->as_dummy_string();
        timer.start(current_node_name);
      }

      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        xs[ai] = &nfxs[arg];
        DYNET_ARG_CHECK(xs[ai]->device == node->device ||
            node->supports_multidevice(),
            "Attempt to do tensor forward in different devices (nodes " <<
            arg << " and " << num_nodes_evaluated << ")");
        ++ai;
      }
      auto& node_fx = nfxs[num_nodes_evaluated];
      node_fx.d = node->dim;
      node_fx.name = "FWD " + node->as_dummy_string();
      // Get the device
      DYNET_ASSERT(node->device != nullptr,
          "Attempt to access null device in "
          "OffloadExecutionEngine::incremental_forward");
      node_fx.device = node->device;
      node_fx.mem_pool = DeviceMempool::FXS;
      // Get the memory to store f(xs)
      auto& node_fx_pools = node_fx.device->pools;
      // If inplaced operation reuse (share) memory and don't call forward
      if(node->forward_inplaced()) {
        // DAO_ERROR("forward inplaced not supported now");
        DYNET_ASSERT(node->args.size() == 1,
                     "Inplacing only supported for arity-1 nodes");
        node_fx.v = nfxs[node->args[0]].v;
      } else {
        // node_fx.v = static_cast<float*>(
        //   node_fx_pools[(int)DeviceMempool::FXS]->allocate(
        //       node->dim.size() * sizeof(float)));
        // cudaMalloc(&node_fx.v, node->dim.size() * sizeof(float));
        // node_fx.v = static_cast<float*>()
        // node_fx.v = offload_allocator->prepare()
        DAO::Kernel kernel;
        for (auto x: xs) kernel.set_inputs(const_cast<TensorUID>(x));
        kernel.set_outputs(&node_fx);
        allocator->prepare(kernel);
        if (node_fx.v == nullptr) {
          DYNET_RUNTIME_ERR("Ran out of memory when executing node " <<
                            num_nodes_evaluated << ", allocating FWD memory.");
        }
        void* aux_mem = nullptr;
        // Is the node requesting extra memory?
        size_t aux_size = node->aux_storage_size();
        if (aux_size) {
          aux_mem = node_fx_pools[(int)DeviceMempool::FXS]->allocate(aux_size);
          if (aux_mem == nullptr)
            DYNET_RUNTIME_ERR("Ran out of auxiliary memory when executing node "
                              << num_nodes_evaluated);
        }
        node->aux_mem = aux_mem;

        // Compute f(xs) and store to node_fx.
        node->forward(xs, node_fx);
        allocator->complete(kernel);
      }

      if (profiling_flag) { timer.stop(current_node_name); }
    }
  }

  return nfxs[i];
}

void OffloadExecutionEngine::backward(bool full) {
  DYNET_ASSERT(nfxs.size() >= cg.nodes.size(),
               "Mismatched array sizes in OffloadExecutionEngine::backward");
  backward((VariableIndex)(cg.nodes.size() - 1), full);
}

void OffloadExecutionEngine::backward(VariableIndex from_where, bool full) {
  if (from_where >= nfxs.size()) { incremental_forward(from_where); }
  if (nfxs[from_where].d.size() != 1) {
    DYNET_INVALID_ARG(
        "backward() can only be called on scalar nodes, but node "
        << from_where << " has dimension: " << nfxs[from_where].d);
  }

  const unsigned num_nodes = from_where + 1;
  ndEdfs.resize(num_nodes);
  const vector<Device*> &devices = device_manager->get_devices();
  for(Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->free();

  // This loop allocates memory on the appropriate devices for the nodes whose
  // derivatives will be computed.
  for (unsigned i = 0; i < num_nodes; ++i) {
    const auto dim = nfxs[i].d;
    auto& node_dEdfx = ndEdfs[i];
    node_dEdfx.d = dim;
    node_dEdfx.device = nfxs[i].device;
    node_dEdfx.mem_pool = DeviceMempool::DEDFS;
    const Node* node = cg.nodes[i];
    // If the operation is inplaced, re-use memory
    if(node->backward_inplaced()) {
      // cerr << node->as_dummy_string() << ", node->args.size() == " << node->args.size() << endl;
      DYNET_ASSERT(node->args.size() == 1,
                   "Inplacing only supported for arity-1 nodes");
      node_dEdfx.v = ndEdfs[node->args[0]].v;
    } else {
      node_dEdfx.v = static_cast<float*>(
          node_dEdfx.device->pools[(int)DeviceMempool::DEDFS]->allocate(
              dim.size() * sizeof(float)));
      if (node_dEdfx.v == nullptr) {
        DYNET_RUNTIME_ERR(
            "out of memory while attempting to allocate space for "
            "derivatives of node " << i << ", allocating BWD memory.");
      }
    }
  }
  // Zero all derivative memory (which is contiguous on each device)
  for (Device* device : devices)
    device->pools[(int)DeviceMempool::DEDFS]->zero_allocated_memory();

  // initialize dE/dE = 1
  ndEdfs.back().v = cg.nodes.back()->device->kSCALAR_ONE;

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
  vector<const Tensor*> xs(16);
  string current_node_name;  // Optionally used for debugging (reused).
  for (int i = num_nodes - 1; i >= 0; --i) {
    if (!in_computation[i]) continue;
    const Node* node = cg.nodes[i];
    // If the operation is inplaced, no need to call backward
    if(node->backward_inplaced()) {
      for (VariableIndex arg : node->args)
        in_computation[arg] = true;
    } else {
      if (profiling_flag) {
        current_node_name = "BWD " + node->as_dummy_string();
        timer.start(current_node_name);
      }
      const auto& node_fx = nfxs[i];  // f(x_1, x_2, ..., x_arity), which
                                      // was previously computed by forward.
      const auto& node_dEdfx = ndEdfs[i];  // dE/df(x_1, x_2, ..., x_arity)
      xs.resize(node->arity());
      unsigned ai = 0;
      for (VariableIndex arg : node->args) {
        in_computation[arg] = true;
        xs[ai] = &nfxs[arg];
        ++ai;
      }
      ai = 0;
      for (VariableIndex arg : node->args) {
        if (needs_derivative[arg]) {
          auto& node_dEdxai = ndEdfs[arg];  // where to store dE/dx_{ai}.
          DYNET_ASSERT(node_fx.device == node_dEdfx.device &&
                       node_fx.device == node_dEdxai.device,
                       "Attempt to do tensor backward in different devices");
          node->backward(xs, node_fx, node_dEdfx, ai, node_dEdxai);
        }
        ++ai;
      }
      if (profiling_flag) { timer.stop(current_node_name); }
    }
  }

  // Accumulate gradients into parameters.
  for (VariableIndex i : cg.parameter_nodes) {
    if (i <= from_where) {
      ParameterNodeBase* pnode = static_cast<ParameterNodeBase*>(cg.nodes[i]);
      pnode->accumulate_grad(ndEdfs[i]);
    }
  }
  backward_computed = from_where+1;
}


} // namespace DAO