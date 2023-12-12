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

struct Kernel {
  std::function<void(Kernel*)> _impl;
  std::vector<TensorUID> _inputs;
  std::vector<TensorUID> _outputs;  
  std::vector<TensorUID> _free_list;
  std::string _name = ""; 
  pid_t _tid = 0;
  bool _stop = false; 

  Kernel& set_impl(std::function<void(Kernel*)> impl) {
    this->_impl = impl;
    return *this; 
  }

  template<typename... Args>
  Kernel& set_inputs(Args... args){
    (_inputs.push_back(args), ...);
    return (*this);
  }

  template<typename... Args>
  Kernel& set_outputs(Args...args) {
    (_outputs.push_back(args), ...);
    return (*this);
  }
  
  template<typename... Args>
  Kernel& set_free(Args...args) {
    (_free_list.push_back(args), ...);
    return (*this);
  }

  Kernel& set_stop() {
    _stop = true;
    return *this;
  }

  bool is_stop() const {
    return _stop;
  }

  Kernel& set_name(const char* name) {
    _name = std::string(name);
    return *this;
  }
}; 

DAO_API void push_kernel(Kernel&& kernel);
DAO_API pid_t get_executor_tid();

} // namespace DAO 
#endif // DAO_KERNEL_QUEUE_H_
