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

    // TODO: MPI init
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
        // TODO: MPI workers
        MPI_Request request;
        int dest = 0;
        MPI_Isend(&local_count, 1, MPI_LONG_LONG, dest, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.

        MPI_Request *requests = (MPI_Request *)malloc((world_size - 1) * sizeof(MPI_Request));
        MPI_Status *status = (MPI_Status *)malloc((world_size - 1) * sizeof(MPI_Status));

        long long *buffer = (long long *)malloc(world_size * sizeof(long long));
        buffer[0] = local_count;

        for(int i = 1; i < world_size; ++i){
            MPI_Irecv(buffer + i, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, requests + i - 1);
        }
        // printf("%d\n", count);

        // MPI_Request requests[];

        MPI_Waitall(world_size - 1, requests, status);

        for(int i = 0; i < world_size; ++i){
            count += buffer[i];
        }

        free(buffer);
        free(requests);
        free(status);
    }

    if (world_rank == 0)
    {
        // TODO: PI result
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
