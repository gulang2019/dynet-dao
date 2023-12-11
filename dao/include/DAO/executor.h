#ifndef _DAO_EXECUTOR_H_
#define _DAO_EXECUTOR_H_

#include <DAO/generator.h>
#include <DAO/utils.h>
#include <DAO/globals.h>

namespace DAO {
class Executor {
public: 
    Executor(ConcurrentQueue<Kernel>& kernel_queue) : kernel_queue_(kernel_queue) {}
    void run();
private: 
    ConcurrentQueue<Kernel>& kernel_queue_;
};

DAO_API void launch();
// void join();
DAO_API void status();
DAO_API void sync();
DAO_API void stop();
DAO_API void log (const char* msg);
DAO_API void initialize(int& argc, char**& argv); 
DAO_API void begin_profile(const char*);
DAO_API void end_profile();

template<typename T>
void complete(const std::shared_ptr<T>& data) {
  if (!async_enabled) return; 
  DAO::Kernel kernel;
  kernel.set_name("complete computation graph");
  kernel.set_impl([data](DAO::Kernel*){}); 
  DAO::push_kernel(std::move(kernel));
}
// void stop();

} // DAO 

#endif // DAO_EXECUTOR_H_
