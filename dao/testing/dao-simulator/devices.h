#pragma once 
#include <math.h>

#include "globals.h"

typedef double event_t;

struct stream_t {
    real_time_t current_time = 0.0;
    real_time_t idle_time = 0.0;
    double bandwidth; 
    void wait_for(event_t event) {
        idle_time += std::max(0.0, event - current_time);
        current_time = std::max(current_time, event);
    }
    void commit(real_time_t time) {
        current_time += time; 
    }
    void copy(size_t size) {    
        current_time += size / bandwidth;
    }
    event_t current() const {
        return current_time;
    }
};