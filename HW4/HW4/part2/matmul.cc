#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#define MASTER 0
#define FROM_MASTER  1
#define FROM_WORKER  2

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    // Allocate contiguous buffers and copy data.
    // Keep A row-major and B column-major (as provided), matching matrix_multiply.


    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //printf("%d\n", rank);
    if(rank == 0){
        const int a_elems = n * m;
        const int b_elems = m * l;

        int *a = new int[a_elems];
        int *b = new int[b_elems];

        memcpy(a, a_mat, a_elems * sizeof(int));
        memcpy(b, b_mat, b_elems * sizeof(int));

        *a_mat_ptr = a;
        *b_mat_ptr = b;
    }

}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_row = 0;
    int rem = 0;
    int offset = 0;
    int rows;

    MPI_Status status;

    //printf("%d\n", rank);
    if(rank == 0){
        local_row = n / (size -1);
        rem = n % (size - 1);
        offset = 0;
        
        for(int dest = 1; dest < size; ++dest){
            rows = (dest <= rem) ? local_row + 1 : local_row;

            MPI_Send(&n, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);

            //printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);

            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows,   1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(a_mat + (offset * m), rows * m, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(b_mat, l * m, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);

            offset += rows;
        }
        for (int i = 1; i < size; i++) {
            int source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows,   1, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(out_mat + (offset * l), rows * l, MPI_INT, source, FROM_WORKER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("Received results from task %d\n", source);
        }

        //memset(out_mat, 0, sizeof(out_mat));
        /*
        printf("****************************************\n");
        printf("Result Matrix C = A x B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++)
                printf("%6.2f ", out_mat[(i * l) + j]);
            printf("\n");
        }
        printf("****************************************\n");
        printf("Done.\n");
        */
    }

    if (rank > 0) {

        int N, M, L;

        MPI_Recv(&N, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&M, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&L, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        //printf("%d %d %d\n", N, M, L);

        MPI_Recv(&offset, 1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows,   1, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int *a = new int[rows * M];
        int *b = new int[L * M];
        int *c = new int[rows * L];

        MPI_Recv(a, rows * M, MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, L * M,  MPI_INT, 0, FROM_MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < L; ++j){
                c[(i * L) + j] = 0;
                for(int k = 0; k < M; ++k){
                    c[(i * L) + j] += a[(i * M) + k] * b[(j * M) + k];
                }
            }
        }
        MPI_Send(&offset, 1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows,   1, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(c, rows * L, MPI_INT, 0, FROM_WORKER, MPI_COMM_WORLD);

        delete[] a;
        delete[] b;
        delete[] c;
    }
}


void destruct_matrices(int *a_mat, int *b_mat)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0){
        delete[] a_mat;
        delete[] b_mat;
    }
}
