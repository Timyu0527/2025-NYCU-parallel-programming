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

    // TODO: init MPI
    int SEED = 12345;
    long long count = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    long long local_tosses = tosses / world_size;
    long long remain = tosses % world_size;
    int seed = (world_rank + 1) * SEED;

    srand(seed);

    if(world_rank <= remain) local_tosses++;
    long long local_count = 0;
    for(int i = 0; i < local_tosses; ++i){
        double x = (2.0 * rand_r(&seed) / RAND_MAX) - 1.0;
        double y = (2.0 * rand_r(&seed) / RAND_MAX) - 1.0;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1) local_count++;
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
        int dest = 0;
        MPI_Send(&local_count, 1, MPI_LONG_LONG, dest, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: main
        // printf("%d\n", count);
        for(int i = 1; i < world_size; ++i){
            long long tmp;
            MPI_Recv(&tmp, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            count += tmp;
        }
        count += local_count;
        // printf("%d\n", count);

    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        double pi_result = 4 * count / ((double)tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
