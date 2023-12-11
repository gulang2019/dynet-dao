#ifndef DAO_INTERFACE_DYNET_H
#define DAO_INTERFACE_DYNET_H

#include <vector>
#include <deque>

#include "DAO/globals.h"

#include "dynet/dynet.h" 
#include "dynet/exec.h"

using dynet::ExecutionEngine;
using dynet::ComputationGraph;
using dynet::VariableIndex;
using dynet::Tensor;

namespace DAO {

class AsyncExecutionEngine: public ExecutionEngine {
 public:
  explicit AsyncExecutionEngine(const ComputationGraph& cg) :
    ExecutionEngine(cg), num_nodes_evaluated(0) {}
  void invalidate() override;
  void invalidate(unsigned i) override;
  const Tensor& forward() override;
  const Tensor& forward(VariableIndex i) override;
  const Tensor& incremental_forward() override;
  const Tensor& incremental_forward(VariableIndex i) override;
  const Tensor& get_value(VariableIndex i) override;
  const Tensor& get_gradient(VariableIndex i) override;
  void backward(bool full = false) override;
  void backward(VariableIndex from_where, bool full = false) override;
public: 
  static logical_time_t logical_time; 
  VariableIndex num_nodes_evaluated;
  VariableIndex backward_computed;
  std::vector<std::shared_ptr<Tensor> > nfxs; 
  std::deque<std::shared_ptr<Tensor> > ndEdfs;
};
/**
 * this function will automatically release the computation graph after 
 * previous computation is done 
*/
void complete(const std::shared_ptr<ComputationGraph>& cg);

} // namespace DAO 

#endif 