#ifndef rtreeH
#define rtreeH
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

// nvcc is C++, not C
#ifdef __cplusplus
extern "C" {
#endif

typedef int key_t;
typedef float ord_t; // ordinate. There's only one, so it's not a coordinate.
#define ord_t_max FLT_MAX;
#define ord_t_lowest -FLT_MAX;

struct rtree_point {
  ord_t x;
  ord_t y;
  key_t key;
};

/// used for optimal cuda x sorting
struct rtree_y_key {
  ord_t y;
  key_t key;
};

/// used for optimal cuda x sorting
struct rtree_points {
  ord_t* x;
  struct rtree_y_key* ykey;
  size_t length;
};

typedef uint64_t location_t;
extern const location_t location_t_max;

struct linear_quadtree {
  location_t*       locations;
  struct rtree_point* points;
  size_t            length;
};

#define LINEAR_QUADTREE_DEPTH (sizeof(location_t) * CHAR_BIT / 2)

struct linear_quadtree lqt_create(struct rtree_point* points, size_t len, 
                                  ord_t xstart, ord_t xend, 
                                  ord_t ystart, ord_t yend,
                                  size_t* depth);
struct linear_quadtree lqt_nodify(struct rtree_point* points, size_t len, 
                                  ord_t xstart, ord_t xend, 
                                  ord_t ystart, ord_t yend,
                                  size_t* depth);
struct linear_quadtree lqt_sortify(struct linear_quadtree);

struct linear_quadtree lqt_create_cuda(struct rtree_point* points, size_t len, 
                                       ord_t xstart, ord_t xend, 
                                       ord_t ystart, ord_t yend,
                                       size_t* depth);
struct linear_quadtree lqt_create_cuda_slow(struct rtree_point* points, size_t len, 
                                            ord_t xstart, ord_t xend, 
                                            ord_t ystart, ord_t yend,
                                            size_t* depth);
struct linear_quadtree lqt_nodify_cuda(struct rtree_point* points, size_t len, 
                                       ord_t xstart, ord_t xend, 
                                       ord_t ystart, ord_t yend,
                                       size_t* depth);
struct linear_quadtree lqt_sortify_cuda(struct linear_quadtree);

void lqt_copy(struct linear_quadtree* destination, struct linear_quadtree* source);
void lqt_delete(struct linear_quadtree);
void lqt_print_node(const location_t* location, const struct rtree_point* point, const bool verbose);
void lqt_print_nodes(struct linear_quadtree lqt, const bool verbose);


#define RTREE_NODE_SIZE 4

struct rtree {
  size_t depth; // needed to know where the leaves begin
  struct rtree_node* root;
};

struct rtree_rect {
  ord_t top;
  ord_t bottom;
  ord_t left;
  ord_t right;
};


struct rtree_leaf {
  struct rtree_rect   bounding_box; ///< MUST be first; see rtree_node_info
  size_t              num;
  struct rtree_point* points;
};

struct rtree_node {
  struct rtree_rect  bounding_box;
  size_t             num;
  struct rtree_node* children;
};

void rtree_print_leaves(const struct rtree_leaf* leaves, const size_t len);
void rtree_print_leaf(const struct rtree_leaf* leaf);
void rtree_print_point(const struct rtree_point* point);

struct rtree_points cuda_sort(struct rtree_points points);
struct rtree_leaf* cuda_create_leaves(struct rtree_points sorted);
struct rtree_node* cuda_create_level(struct rtree_node* nodes, const size_t nodes_len);
struct rtree_node* cuda_create_level_leaf(struct rtree_leaf* nodes); ///< @todo use macro
struct rtree cuda_create_rtree(struct rtree_points points);

void rtree_print_rect(struct rtree_rect r);
void rtree_print_node(struct rtree_node* n, const size_t depth);
void rtree_print(struct rtree tree);

#ifdef __cplusplus
}
#endif

struct linear_quadtree_cuda {
  struct rtree_point* points;
  location_t*       cuda_locations;
  struct rtree_point* cuda_points;
  size_t            length;
};
struct linear_quadtree_cuda lqt_nodify_cuda_mem(struct rtree_point* points, size_t len, 
                                                ord_t xstart, ord_t xend, 
                                                ord_t ystart, ord_t yend,
                                                size_t* depth);
struct linear_quadtree lqt_sortify_cuda_mem(struct linear_quadtree_cuda);



#endif
