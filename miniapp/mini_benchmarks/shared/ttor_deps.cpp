#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include "tasktorrent/tasktorrent.hpp"
#include "../common.hpp"

typedef std::array<int,2> int2;

/**
 * n_rows rows of tasks over n_cols columns
 * with n_edges deps between [i,j] and [i+k,j+1] for 0 <= k < n_edges
 */
int wait_chain_deps(const int n_threads, 
                    const int n_rows, 
                    const int n_edges, 
                    const int n_cols, 
                    const double spin_time, 
                    const int repeat, 
                    const int verb) {

    std::vector<double> efficiencies;
    std::vector<double> times;
    int n_tasks = n_rows * n_cols;

    for(int step = 0; step < repeat; step++) {

        ttor::Threadpool_shared tp(n_threads, verb <= 1 ? 0 : verb, "Wk_");
        ttor::Taskflow<int2> tf(&tp, verb <= 1 ? 0 : verb);
        std::atomic<size_t> n_tasks_ran(0);

        tf.set_mapping([&](int2 ij) {
            return (ij[0] % n_threads);
        })
        .set_indegree([&](int2 ij) {
            return (ij[1] == 0 ? 1 : n_edges);
        })
        .set_task([&](int2 ij) {
            n_tasks_ran++;
            spin_for_seconds(spin_time);
            if(ij[1] < n_cols-1) {
                for(int k = 0; k < n_edges; k++) {
                    tf.fulfill_promise({ (ij[0] + k) % n_rows, ij[1]+1 });
                }
            }
        })
        .set_priority([&](int2 ij) {
            return n_cols - (double)ij[1];
        })
        .set_name([&](int2 ij){return std::to_string(ij[0]) + "_" + std::to_string(ij[1]);});

        auto t0 = ttor::wctime();
        for(int k = 0; k < n_rows; k++) {
            tf.fulfill_promise({k,0});
        }
        tp.join();
        auto t1 = ttor::wctime();
        double time = ttor::elapsed(t0, t1);
        if(verb) printf("iteration repeat n_threads n_rows n_edges n_cols spin_time time n_tasks efficiency\n");
        assert(n_tasks_ran.load() == n_tasks);
        double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        double efficiency = speedup / (double)(n_threads);
        efficiencies.push_back(efficiency);
        times.push_back(time);
        printf("++++ ttordeps %d %d %d %d %d %d %e %e %d %e\n", step, repeat, n_threads, n_rows, n_edges, n_cols, spin_time, time, n_tasks, efficiency);

    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    if(verb) printf("repeat n_threads n_rows n_edges n_cols spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf(">>>> ttordeps %d %d %d %d %d %e %d %e %e %e %e\n", repeat, n_threads, n_rows, n_edges, n_cols, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);

    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_rows = 10;
    int n_edges = 10;
    int n_cols = 5;
    double spin_time = 1e-6;
    int verb = 0;
    int repeat = 1;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        if(n_threads <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        n_rows = atoi(argv[2]);
        if(n_rows <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        n_edges = atoi(argv[3]);
        if(n_edges <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        n_cols = atoi(argv[4]);
        if(n_cols <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 6)
    {
        spin_time = atof(argv[5]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 7)
    {
        repeat = atof(argv[6]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 8)
    {
        verb = atoi(argv[7]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("./ttor_deps n_threads n_rows n_edges n_cols spin_time repeat verb\n");
    int error = wait_chain_deps(n_threads, n_rows, n_edges, n_cols, spin_time, repeat, verb);

    return error;
}
