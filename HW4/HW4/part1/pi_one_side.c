#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    int SEED = 12345;
    long long *count;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long local_tosses = tosses / world_size;
    long long remain = tosses % world_size;
    int seed = (world_rank + 1) * SEED;

    srand(seed);


    if(world_rank < remain) local_tosses++;
    long long local_count = 0;

    for(int i = 0; i < local_tosses; ++i){
        double x = (2.0 * rand_r(&seed) / RAND_MAX) - 1.0;
        double y = (2.0 * rand_r(&seed) / RAND_MAX) - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) local_count++;
    }

    int size = world_size;
    int rank = world_rank;

    long long *tmp = (long long *)malloc(size * sizeof(long long));

    if (world_rank == 0)
    {
        // Main
        MPI_Alloc_mem(sizeof(long long), MPI_INFO_NULL, &count);
        *count = local_count;
        MPI_Win_create(count, sizeof(long long), sizeof(long long), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&local_count, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        double pi_result = 4.0 * (*count) / ((double)tosses);
        MPI_Free_mem(count);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
