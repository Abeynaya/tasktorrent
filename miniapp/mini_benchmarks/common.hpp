#ifndef __TTOR_TESTS_COMMON__
#define __TTOR_TESTS_COMMON__

#include <chrono>
#include <numeric>
#include <cmath>
#include <vector>
#include <array>

typedef std::array<int,2> int2;

std::chrono::time_point<std::chrono::high_resolution_clock> wtime_now() {
    return std::chrono::high_resolution_clock::now();
};

double wtime_elapsed(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0, const std::chrono::time_point<std::chrono::high_resolution_clock>& t1) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
};

void spin_for_seconds(double time) {
    auto t0 = std::chrono::high_resolution_clock::now();
    while(true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        if( elapsed_time >= time ) break;
    }
}

void compute_stats(const std::vector<double> &data, double *average, double* std) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double stddev = 0.0;
    for(int k = 0; k < data.size(); k++) {
        stddev += (data[k] - mean) * (data[k] - mean);
    }
    stddev = std::sqrt(stddev / (data.size() - 1.0));
    
    *average = mean;
    *std = stddev;
}

#endif
