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
#include <chrono>

#include <DAO/globals.h>
#include <stack>

namespace DAO {

struct TimerValue
  {
    std::map<std::string, double> values;
    std::map<std::string, int> int_values;
    std::map<std::string, std::vector<int>> log_values;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
  };

  struct Timer
  {
    std::string type;
    std::string scope_tag;
    std::stack<std::string> last_key;
    Timer(const char *t = "ADD");
    void start(const std::string& key);
    void stop(const std::string& key);
    void stop();
    void scope(std::string key);
    void lock();
    void unlock();

    void show(std::ostream& o = std::cout) const;
    void clear();
    void clearall();

    void cumint(std::string key, int v);

    void log(std::string key, int v);

    void save(const std::string& filename) const;

    double get(std::string key);

    std::unordered_map<std::string, TimerValue> scopes;

    bool locked = false;
  };

} // namespace DAO 
#endif // DAO_UTIL_H_
