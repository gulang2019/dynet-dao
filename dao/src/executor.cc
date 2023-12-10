#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <DAO/executor.h>
#include <DAO/generator.h>
#include <DAO/globals.h>
#include <DAO/utils.h>

#include <cuda_runtime.h>
#include <nvToolsExt.h>

namespace DAO {

extern ConcurrentQueue<Kernel> kernel_queue;
extern ConcurrentCounter kernel_counter;
extern ConcurrentValue<pid_t> executor_tid;

static std::thread executor_thread;

void Executor::run() {
  while (true) {    
    Kernel kernel = kernel_queue_.pop();
    DAO_INFO("Executor: %s, %d, %d in kernel_queue", kernel._name.c_str(), kernel._tid, kernel_queue.size());
    if (kernel.is_stop()) {
      DAO_INFO("Executor::run(): stop kernel");
      if (kernel_counter.peek()!=0) {
        DAO_WARNING("Executor: stop when kernel_counter is not zero");
        kernel_counter.set_zero();
      }
      break;
    }
    kernel._impl(&kernel); 
    kernel_counter.decrement();
  }
}

static bool launched = false;

void status() {
  DAO_INFO("status: launched %d kernel_counter(%p) = %d, %d in kernel_queue(%p)", 
    launched, &kernel_counter, kernel_counter.peek(), kernel_queue.size(), &kernel_queue);
}

void sync() {
  if (!async_enabled) return;
  DAO_INFO("DAO::sync");
  status();
  if (launched) kernel_counter.wait_until_zero();
  cudaDeviceSynchronize();
}

void launch(){
  if (!async_enabled) return;
  if (launched) {
    DAO_WARNING("executor has already been launched");
    return;
  }
  launched = true;
  DAO_INFO("launching kernel_queue address = %p", &kernel_queue);
  auto _entry = [](){
    DAO::executor_tid.set(gettid());
    Executor executor(kernel_queue); 
    executor.run();
  };
  executor_thread = std::thread(_entry);
  executor_tid.wait_until_has_value();
  executor_thread.detach();
}

void stop() {
  if (!async_enabled) return;
  DAO_INFO("DAO::stop");
  status();
  Kernel kernel;
  kernel.set_stop().set_name("stop");
  kernel_queue.push(std::move(kernel));
}

void log(const char* msg) {
  Kernel kernel;
  kernel.set_name(msg).set_impl([](Kernel* kernel){
    printf(ANSI_COLOR_BLUE "[DAO::Kernel Log]: %s\n" ANSI_COLOR_RESET, kernel->_name.c_str()); 
  });
  kernel_counter.increment();
  kernel_queue.push(std::move(kernel));
}

void initialize(int& argc, char**& argv) {
  int new_argc = 1;
  int i = 1;
  while(i < argc) {
      if (strcmp(argv[i], "--dao-disable") == 0) {
          async_enabled = false; 
      } else if (strcmp(argv[i], "--dao-verbose") == 0) {
          verbose = std::stoi(argv[i+1]);
          i += 1;
      }
      else {
        argv[new_argc++] = argv[i];
      }
      i++; 
  }
  argc = new_argc;
  if (async_enabled)
      launch();
}

void begin_profile(const char*name){
  if (!async_enabled) return;
  Kernel kernel;
  kernel.set_name(name).set_impl([](Kernel* kernel){
    nvtxRangePush(kernel->_name.c_str());
  });
  DAO::push_kernel(std::move(kernel));
}

void end_profile() {
  if (!async_enabled) return;
  Kernel kernel;
  kernel.set_name("end_profile").set_impl([](Kernel* kernel){
    nvtxRangePop();
  });
  DAO::push_kernel(std::move(kernel));
}

} // DAO 