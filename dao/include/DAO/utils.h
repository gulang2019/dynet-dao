// concurrent-queue.h
#ifndef DAO_UTIL_H_
#define DAO_UTIL_H_

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <functional>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>

#include <DAO/globals.h>

namespace DAO {

template <typename T>
class ConcurrentQueue {
 public:
  T pop() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
    return val;
  }

  void pop(T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty()) {
      // DAO_INFO("ConcurrentQueue:pop(): queue is empty, wait");
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
    mlock.unlock();
    cond_.notify_one();
  }

  void wait_until_empty() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (!queue_.empty()) {
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }

  void push(const T& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
      // DAO_INFO("ConcurrentQueue::push(): queue is full, wait");
       cond_.wait(mlock);
    }
    DAO_INFO("push kernel: %s, tid %d", item._name.c_str(), gettid());
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }

  void push(T&& item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.size() >= BUFFER_SIZE) {
      // DAO_INFO("ConcurrentQueue::push(): queue is full, wait");
       cond_.wait(mlock);
    }
    DAO_INFO("push kernel: %s, tid %d", item._name.c_str(), gettid());
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  
  // is this thread safe? 
  int size() const {
    return queue_.size();
  }

  ConcurrentQueue() = default;
  ConcurrentQueue(const ConcurrentQueue&) = delete;            // disable copying
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete; // disable assignment
  ~ConcurrentQueue() = default;
 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  const static unsigned int BUFFER_SIZE = 1000;
};

class ConcurrentCounter {
public: 
  void increment() {
    // DAO_INFO("Wait Incrementing");
    std::unique_lock<std::mutex> mlock(mutex_);
    // DAO_INFO("Incrementing counter %d", count_);
    count_++;
    mlock.unlock();
    cond_.notify_one();
  }
  void decrement() {
    // DAO_INFO("Wait Decrement");
    std::unique_lock<std::mutex> mlock(mutex_);
    // DAO_INFO("Decrementing counter %d", count_);
    count_--;
    mlock.unlock();
    cond_.notify_one();
  }
  void set_zero() {
    // DAO_INFO("Wait Decrement");
    std::unique_lock<std::mutex> mlock(mutex_);
    // DAO_INFO("Decrementing counter %d", count_);
    count_ = 0;
    mlock.unlock();
    cond_.notify_one();
  }
  void wait_until_zero() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (count_ != 0) {
      // DAO_INFO("Wait until zero %d", count_);
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }
  int peek() const {
    return count_;
  }
private: 
  std::condition_variable cond_;
  std::mutex mutex_; 
  int count_ = 0; 
};

template<typename T>
class ConcurrentValue {
public: 
  void set(T value) {
    std::unique_lock<std::mutex> mlock(mutex_);
    value_ = value;
    has_value_ = true;
    mlock.unlock();
    cond_.notify_one();
  }
  T get() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (!has_value_) {
      cond_.wait(mlock);
    }
    auto val = value_;
    mlock.unlock();
    cond_.notify_one();
    return val;
  }
  void wait_until_has_value() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (!has_value_) {
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }
  void wait_until_no_value() {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (has_value_) {
      cond_.wait(mlock);
    }
    mlock.unlock();
    cond_.notify_one();
  }
  ConcurrentValue() = default;
  ConcurrentValue(const ConcurrentValue&) = delete;            // disable copying
  ConcurrentValue& operator=(const ConcurrentValue&) = delete; // disable assignment
  ~ConcurrentValue() = default;
private: 
  bool has_value_ = false;
  T value_;
  std::condition_variable cond_;
  std::mutex mutex_; 
}; 

} // namespace DAO 
#endif // DAO_UTIL_H_
