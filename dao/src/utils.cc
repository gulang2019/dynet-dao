#include <DAO/utils.h>

#include <fstream>
#include <vector>
#include <string>

using std::string;
using std::vector; 
using std::pair;

namespace DAO {

Timer::Timer(const char *name) : _name(name)
  {
    scope_tag = "default";
    scopes[scope_tag] = {};
  }
  void Timer::start(const std::string& key)
  {
    if (locked)
      return;
    auto &scope = scopes[scope_tag];
    scope.start_times[key] = std::chrono::high_resolution_clock::now();
    scope.keys.push_back(key);
  }

void Timer::stop(const std::string& key)
  {
    auto &scope = scopes[scope_tag];
    if (locked)
      return;
    if (!scope.start_times.count(key))
      return;
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - scope.start_times[key]).count();
    if (scope.values.count(key) == 0)
      scope.values[key] = 0.0;
    scope.values[key] += elapsed;
    scope.start_times.erase(key);
    if (auto it = std::find(scope.keys.begin(), scope.keys.end(), key); it != scope.keys.end())
      scope.keys.erase(it);
    return;
  }

  std::string Timer::stop() {
    auto& scope = scopes[scope_tag];
    std::string ret = scope.keys.back();
    Timer::stop(scope.keys.back());
    return ret;
  }

  void Timer::lock() { locked = true; }
  void Timer::unlock() { locked = false; }

  // void Timer::show(std::string label, std::initializer_list<std::string> scope_tags, int repeat)
  // {
  //   show(label, std::vector<std::string>(scope_tags), repeat);
  // }

  void Timer::show(std::ostream& o) const
{
    std::string scope_name = "default";
    auto &scope = scopes.at(scope_name);
    vector<pair<string, double>> times;
    for (auto kv : scope.values)
        times.push_back(kv);
    sort(times.begin(), times.end(), [](pair<string, double> &v1, pair<string, double> &v2)
        { return v1.second < v2.second; });
    for (auto kv : times)
    {
        o << "\t" << kv.first << ":\t" << kv.second << "ms\n";
    }
    for (auto kv : scope.int_values)
    {
        o << "\t" << kv.first << ":\t" << (kv.second >> 20) << "M" << "\n";
    }
    for (auto kv: scope.log_values) {
        o << "\t" << kv.first << ":";
        for (auto v:kv.second) o << v << ",";
        o << "\n";
    }
}
  void Timer::clear()
  {
    auto &scope = scopes[scope_tag];
    scope.values.clear();
    scope.start_times.clear();
    scope.log_values.clear();
    scope.int_values.clear();
  }

  void Timer::clearall(){
    scopes.clear();
    scopes["default"] = {};
    scope_tag = "default";
  }

  void Timer::cumint(std::string key, size_t v)
  {
    auto &scope = scopes[scope_tag];
    if (locked)
      return;
    if (scope.int_values.count(key) == 0)
    {
      scope.int_values[key] = 0;
    }
    scope.int_values[key] += v;
  }

  void Timer::log(std::string key, int v)
  {
    if (locked)
      return;
    auto &scope = scopes[scope_tag];
    if (scope.log_values.count(key) == 0)
      scope.log_values[key] = {};
    scope.log_values[key].push_back(v);
  }

  void Timer::save(const std::string& filename) const
  {
    std::string prefix; 
    std::string _filename;
    size_t lastDotPosition = filename.find_last_of('.');
    if (lastDotPosition != std::string::npos) {
        prefix = filename.substr(0, lastDotPosition);
    }
    else prefix = filename;
    _filename += prefix + "." + _name + ".time.csv";
    std::ofstream file(_filename);
    auto& scope = scopes.at("default");
    file << "metric,value,percent" << std::endl;
    std::vector<std::pair<std::string, double>> values;
    double tot = 0;
    for (auto kv : scope.values) {
      values.push_back(kv);
      tot += kv.second;
    }
    sort(values.begin(), values.end(), [](std::pair<std::string, double> &v1, std::pair<std::string, double> &v2)
        { return v1.second > v2.second; });
    for (auto kv : values) {
      file << kv.first << "," << kv.second << "," << kv.second/tot*100 << std::endl;
    }
    file.close();

    _filename = prefix + "." + _name + ".profiling.csv";
    file.open(_filename);
    file << "metric,value" << std::endl;
    for (auto kv: scope.int_values)
    {
      file << kv.first << "," << (kv.second >> 20) << std::endl;
    }
    file.close();
  }

  double Timer::get(std::string key)
  {
    auto scope = scopes[scope_tag];
    if (scope.values.count(key) == 0)
      return 0.0f;
    return scope.values[key];
  }

  void Timer::scope(std::string key){
    scope_tag = key;
  }

} // DAO 