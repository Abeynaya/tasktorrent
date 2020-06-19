#include <vector>
#include <memory>
#include <iostream>
#include <cstdlib>
#include "common.hpp"
#include <starpu.h>

/** 
 * Compile with something like
 * icpc -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib  -lstarpu-1.3 -lhwloc starpu_wait.cpp -O3 -o starpu_wait -Wall
 */

double SPIN_TIME = 0.0;

void task(void *buffers[], void *cl_arg) { 
    spin_for_seconds(SPIN_TIME);
}

struct starpu_codelet task_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { task, NULL },
    .nbuffers = 0,
    .modes = { }
};

int wait_only(const int n_tasks, const double spin_time, const int repeat, const int verb) {

    SPIN_TIME = spin_time;
    
    const char* env_n_cores = std::getenv("STARPU_NCPU");
    if(env_n_cores == nullptr) { printf("Missing STARPU_NCPU\n"); exit(1); }
    const int n_threads = atoi(env_n_cores);

    std::vector<double> efficiencies;
    std::vector<double> times;
    for(int step = 0; step < repeat; step++) {

        int err = starpu_init(NULL);
        if(err != 0) { printf("Error in starpu_init!\n"); exit(1); }
        const auto t0 = wtime_now();
        for (int k = 0; k < n_tasks; k++) {
            starpu_task_insert(&task_cl, 0);
        }
        starpu_task_wait_for_all();
        const auto t1 = wtime_now();
        starpu_shutdown();

        const auto time = wtime_elapsed(t0, t1);
        if(verb) printf("iteration repeat n_threads n_tasks spin_time time efficiency\n");
        const double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        const double efficiency = speedup / (double)(n_threads);
        times.push_back(time);
        efficiencies.push_back(efficiency);
        printf("++++ starpu %d %d %d %d %e %e %e\n", step, repeat, n_threads, n_tasks, spin_time, time, efficiency);

    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    if(verb) printf("repeat n_threads spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf(">>>> starpu %d %d %e %d %e %e %e %e\n", repeat, n_threads, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);

    return 0;
}

int main(int argc, char **argv)
{
    int n_tasks = 1000;
    double spin_time = 1e-6;
    int repeat = 1;
    int verb = 0;

    if (argc >= 2)
    {
        n_tasks = atoi(argv[1]);
        if(n_tasks < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        spin_time = atof(argv[2]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        repeat = atof(argv[3]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        verb = atoi(argv[4]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("STARPU_NCPU=XX ./starpu_wait n_tasks spin_time verb\n");
    int error = wait_only(n_tasks, spin_time, repeat, verb);
    return error;
}
