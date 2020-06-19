#include <starpu.h>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <atomic>

#include "common.hpp"

/** 
 * Compile with something like
 * icpc -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib  -lstarpu-1.3 -lhwloc starpu_deps.cpp -O3 -o starpu_deps -Wall
 */

// TODO: fix global variable here
double SPIN_TIME = 0.0;
int N_DEPS = 0;
std::atomic<size_t> n_tasks_ran(0);
std::vector<bool> done;

struct params
{
    int i;
    int j;
};

void task(void *buffers[], void *cl_arg) { 
    n_tasks_ran++;
    struct params *params = (struct params *)cl_arg;
    // printf("Task %d %d\n", params->i, params->j);
    spin_for_seconds(SPIN_TIME);
}

struct starpu_codelet task_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { task, NULL },
    .nbuffers = 0,
    .modes = { }
};

int wait_chain_deps(const int n_rows, 
                    const int n_edges, 
                    const int n_cols, 
                    const double spin_time, 
                    const int repeat, 
                    const int verb) {

    std::vector<double> efficiencies;
    std::vector<double> times;
    int n_tasks = n_rows * n_cols;

    const char* env_n_cores = std::getenv("STARPU_NCPU");
    assert(env_n_cores != nullptr);
    const int n_threads = atoi(env_n_cores);
    
    SPIN_TIME = spin_time;

    for(int step = 0; step < repeat; step++) {

        n_tasks_ran.store(0);
        int err = starpu_init(NULL);
        assert(err == 0);
        auto tasks = std::vector<struct starpu_task*>(n_rows * n_cols);

        double start = starpu_timing_now();
        for(int j = 0; j < n_cols; j++) {
            for(int i = 0; i < n_rows; i++) {
                tasks[i + j * n_rows] = starpu_task_build(&task_cl, 0);
                struct params *params = (struct params*) malloc(sizeof(struct params));
                params->i = i;
                params->j = j;
                tasks[i + j * n_rows]->cl_arg = params;
                assert(tasks[i + j * n_rows] != nullptr);
                int n_deps = (j == 0 ? 0 : n_edges);
                struct starpu_task** deps = (struct starpu_task**) malloc(n_deps * sizeof(struct starpu_task*));
                for(int k = 0; k < n_deps; k++) {
                    // In-deps, wrapping around
                    // It's just a reverse modulo, basically
                    int i_before = n_rows - ( ( (n_rows - i - 1) + k ) % n_rows ) - 1;
                    deps[k] = tasks[i_before + (j - 1) * n_rows];
                    assert(i_before >= 0 && i_before < n_rows);
                }
                starpu_task_declare_deps_array(tasks[i + j * n_rows], n_deps, deps);
            }
            // Can only submit when we're done using them as dependencies
            if(j > 0) {
                for(int i = 0; i < n_rows; i++) {
                    int err = starpu_task_submit(tasks[i + (j - 1) * n_rows]);
                    assert(err == 0);
                }
            }
        }
        // Submit the last column of tasks
        for(int i = 0; i < n_rows; i++) {
            int err = starpu_task_submit(tasks[i + (n_cols - 1) * n_rows]);
            assert(err == 0);
        }
        starpu_task_wait_for_all();
        double end = starpu_timing_now();
        starpu_shutdown();
        double time = (end - start)/1e6;
        if(verb) printf("iteration repeat n_threads n_rows n_edges n_cols spin_time time n_tasks efficiency\n");
        assert(n_tasks_ran.load() == n_tasks);
        double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        double efficiency = speedup / (double)(n_threads);
        efficiencies.push_back(efficiency);
        times.push_back(time);
        printf("++++ starpudeps %d %d %d %d %d %d %e %e %d %e\n", step, repeat, n_threads, n_rows, n_edges, n_cols, spin_time, time, n_tasks, efficiency);
    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    if(verb) printf("repeat n_threads n_rows n_edges n_cols spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf(">>>> starpudeps %d %d %d %d %d %e %d %e %e %e %e\n", repeat, n_threads, n_rows, n_edges, n_cols, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);

    return 0;
}

int main(int argc, char **argv)
{
    int n_rows = 10;
    int n_edges = 10;
    int n_cols = 5;
    double spin_time = 1e-6;
    int repeat = 1;
    int verb = 0;

    if (argc >= 2)
    {
        n_rows = atoi(argv[1]);
        if(n_rows <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        n_edges = atoi(argv[2]);
        if(n_edges <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        n_cols = atoi(argv[3]);
        if(n_cols <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        spin_time = atof(argv[4]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 6)
    {
        repeat = atof(argv[5]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 7)
    {
        verb = atoi(argv[6]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("./starpu_deps n_rows n_edges n_cols spin_time repeat verb\n");
    int error = wait_chain_deps(n_rows, n_edges, n_cols, spin_time, repeat, verb);

    return error;
}
