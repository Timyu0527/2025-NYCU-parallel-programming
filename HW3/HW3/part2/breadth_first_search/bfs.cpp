#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <cstring>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    std::vector<Vertex> next_partial[20];
    int thread_count = omp_get_max_threads();
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                int thread_id = omp_get_thread_num();
                distances[outgoing] = distances[node] + 1;
                next_partial[thread_id].push_back(outgoing);
            }
        }
    }

    for(int i = 0; i < thread_count; ++i){
        for(Vertex j:next_partial[i]){
            int index = new_frontier->count++;
            new_frontier->vertices[index] = j;
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
void bottom_up_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances, int level)
{
    std::vector<Vertex> next_partial[20];
    int thread_count = omp_get_max_threads();
    int n = g->num_nodes;
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,2048)
        for (int i = 0; i < n; i++)
        {
            int node = i;
            if (distances[node] == NOT_VISITED_MARKER){

                const Vertex *start= incoming_begin(g, node);
                const Vertex *end= incoming_end(g, node);

                // attempt to add all neighbors to the new frontier
                for (const Vertex *parent = start; parent != end; parent++)
                {
                    int incoming = *parent;

                    if (distances[incoming] == level)
                    {
                        int thread_id = omp_get_thread_num();
                        distances[node] = level + 1;
                        next_partial[thread_id].push_back(node);
                        break;
                    }
                }
            }

        }
    }

    std::vector<int> sizes(thread_count), offs(thread_count + 1);
    #pragma omp parallel for
    for (int t = 0; t < thread_count; ++t) sizes[t] = (int)next_partial[t].size();

    offs[0] = 0;
    for (int t = 0; t < thread_count; ++t) offs[t + 1] = offs[t] + sizes[t];
    const int total = offs.back();

    new_frontier->count = total;

    #pragma omp parallel for
    for (int t = 0; t < thread_count; ++t) {
        std::memcpy(new_frontier->vertices + offs[t],
                    next_partial[t].data(),
                    (size_t)sizes[t] * sizeof(Vertex));
    }
    /*
    for(int i = 0; i < thread_count; ++i){
        #pragma omp parallel for
        for(int j = 0; j < next_partial[i].size(); ++j){
            new_frontier->vertices[new_frontier->count + j] = next_partial[i][j];
        }
        new_frontier->count += next_partial[i].size();
    }
    */
}
void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        bottom_up_step(graph, frontier, new_frontier, sol->distances, level);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        level++;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
    /*
    int n = graph->num_nodes;
    std::vector<Vertex> frontier(n, 0), next_frontier(n, 0);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier[ROOT_NODE_ID] = 1;
    sol->distances[ROOT_NODE_ID] = 0;

    for (int level = 0; ; ++level) {

        int any_add = 0;  // 用 int 做 OR reduction，比 bool 較通用
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            // 只處理「尚未拜訪」的節點（Bottom-Up 精髓）
            if (sol->distances[i] != NOT_VISITED_MARKER) continue;

            const Vertex* start = incoming_begin(graph, i);
            const Vertex* end = incoming_end(graph, i);

            // 檢查是否「有任一鄰居在 frontier」
            for (const Vertex* j = start; j != end; ++j) {
                if (frontier[*j]) {
                    next_frontier[i] = 1;          // 標記進下一層
                    sol->distances[i] = sol->distances[*j] + 1;  // 直接用 level+1 最穩
                    any_add = 1;          // 本輪有擴張
                    break;
                }
            }
        }

        if (!any_add) break;               // 沒新節點 → 完成
        frontier.swap(next_frontier);            // 下一輪
        memset(next_frontier.data(), 0, n);
    }
    */
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    int alpha = 16, beta = 18, n = graph->num_nodes, m = graph->num_edges;

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    
    int level = 0;
    bool bottom_up = 0;
    double threshold = 0.2 * (double)n;

    while(frontier->count != 0){
        /*
        long long frontier_edges = 0;
        #pragma omp parallel for reduction(+:frontier_edges)
        for (int i=0; i<frontier->count; ++i) {
          Vertex u = frontier->vertices[i];
          frontier_edges += (outgoing_end(graph,u) - outgoing_begin(graph,u));
        }

        if (!bottom_up && frontier_edges > m / alpha) bottom_up = true;
        if (bottom_up && frontier->count < n / beta) bottom_up = false;
        */
        if(frontier->count > threshold) bottom_up = 1;
        else bottom_up = 0;

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        if (bottom_up) {
          bottom_up_step(graph, frontier, new_frontier, sol->distances, level);
        } else {
          top_down_step(graph, frontier, new_frontier, sol->distances);
        }

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;

        level++;
    }
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}
