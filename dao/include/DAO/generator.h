// concurrent-queue.h
#ifndef DAO_KERNEL_QUEUE_H_
#define DAO_KERNEL_QUEUE_H_

#include <vector> 
#include <functional> 
#include <optional>
#include <string> 
#include <unistd.h>

// #include <c10/util/Optional.h>

#include <DAO/globals.h>

namespace DAO {

void push(std::vector<dynet::Tensor*>& vec, const std::vector<const dynet::Tensor*>& to_push);

void push(std::vector<TensorUID>& vec, const std::vector<TensorUID>& to_push); 

void push(std::vector<TensorUID>& vec, TensorUID to_push);

struct Kernel {
  std::function<void(Kernel*)> _impl;
  std::vector<TensorUID> _inputs;
  std::vector<TensorUID> _outputs;  
  std::vector<TensorUID> _free_list;
  std::vector<TensorUID> _zeroed;
  std::string _name = ""; 
  pid_t _tid = 0;

  Kernel&& set_impl(std::function<void(Kernel*)> impl) {
    this->_impl = impl;
    return std::move(*this); 
  }

  template<typename... Args>
  Kernel&& set_inputs(Args... args){
    (push(_inputs, args), ...);
    return std::move(*this);
  }

  template<typename... Args>
  Kernel&& set_outputs(Args...args) {
    (push(_outputs, args), ...);
    return std::move(*this);
  }
  
  template<typename... Args>
  Kernel&& set_zeroed(Args...args) {
    (push(_zeroed, args), ...);
    return std::move(*this);
  }

  Kernel&& set_name(const char* name) {
    _name = std::string(name);
    return std::move(*this);
  }
}; 

} // namespace DAO 
#endif // DAO_KERNEL_QUEUE_H_
