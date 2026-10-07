#define _GNU_SOURCE

#include <sched.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count < 1) {
        perror("sysconf");
        return 1;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);
    if (sched_setaffinity(0, sizeof(set), &set) == -1) {
        perror("sched_setaffinity");
        return 1;
    }

    int current_cpu = sched_getcpu();
    if (current_cpu == -1) {
        perror("sched_getcpu");
        return 1;
    }

    printf("Total online CPUs: %ld\n", cpu_count);
    printf("Bound to CPU: %d\n", current_cpu);
    return 0;
}
