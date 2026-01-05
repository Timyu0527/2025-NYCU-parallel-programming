#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <cstring>
#include <cstdio>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{

    // initialize vertex weights to uniform probability. Double
    // precision scores are used to avoid underflow for large graphs

    int nnodes = num_nodes(g);
    double equal_prob = 1.0 / nnodes;
    for (int i = 0; i < nnodes; ++i)
    {
        solution[i] = equal_prob;
    }

    // For PP students: Implement the page rank algorithm here.  You
    // are expected to parallelize the algorithm using openMP.  Your
    // solution may need to allocate (and free) temporary arrays.

    // Basic page rank pseudocode is provided below to get you started:

    // initialization: see example code above
    double *score_old = (double *)calloc(nnodes, sizeof(double));
    double *score_new = (double *)calloc(nnodes, sizeof(double));
    memcpy(score_old, solution, nnodes * sizeof(double));


    bool converged = 0;
    while (!converged)
    {
        double no_outgoing_score = 0;
        double global_diff = 0.0;
        #pragma omp parallel
        {
            #pragma omp for reduction(+:no_outgoing_score)
            for (Vertex vj = 0; vj < nnodes; ++vj)
            {
                if(outgoing_size(g, vj) == 0)
                {
                    no_outgoing_score += damping * score_old[vj] / nnodes;
                }
            }
            #pragma omp for reduction(+:global_diff)
            for (Vertex vi = 0; vi < nnodes; ++vi)
            {
                const Vertex *start = incoming_begin(g, vi);
                const Vertex *end = incoming_end(g, vi);
                double score = 0.0;
                for (const Vertex *vj = start; vj != end; ++vj)
                {
                    score += score_old[*vj] / outgoing_size(g, *vj);
                }

                score_new[vi] = (damping * score) + (1.0 - damping) / nnodes + no_outgoing_score;

                global_diff += fabs(score_new[vi] - score_old[vi]);
            }

        }
        // compute how much per-node scores have changed
        // quit once algorithm has converged

        converged = (global_diff < convergence);

        memcpy(score_old, score_new, nnodes * sizeof(double));
        //printf("%f %f\n", convergence, global_diff);
    }
    memcpy(solution, score_new, nnodes * sizeof(double));
    free(score_old);
    free(score_new);
}
