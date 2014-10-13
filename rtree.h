#ifndef rtreeH
#define rtreeH
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>

// efficient ceil((float)a/(float)b)
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

// used for optimal cuda x sorting
struct rtree_points {
  ord_t* x;
  struct rtree_y_key* ykey;
  size_t length;
};


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
  struct rtree_rect   bounding_box; ///< MUST be first, so leaf boundary can be checked as node
  size_t              num;
  struct rtree_point* points;
};

struct rtree_node {
  struct rtree_rect  bounding_box; ///< MUST be first, so leaf boundary can be checked as node
  size_t             num;
  struct rtree_node* children;
};

void rtree_print_leaves(const struct rtree_leaf* leaves, const size_t len);
void rtree_print_leaf(const struct rtree_leaf* leaf);
void rtree_print_point(const struct rtree_point* point);

struct rtree_points cuda_sort(struct rtree_points points);
struct rtree_leaf* cuda_create_leaves(struct rtree_points sorted);
struct rtree_node* cuda_create_level(struct rtree_node* nodes, const size_t nodes_len);
struct rtree cuda_create_rtree(struct rtree_points points);

struct rtree cuda_create_rtree_heterogeneously(struct rtree_point* points, const size_t len, const size_t threads);
struct rtree_point* tbb_sort(struct rtree_point* points, const size_t len, const size_t threads);
struct rtree_leaf* cuda_create_leaves_together(struct rtree_point* sorted, const size_t len);

struct rtree cuda_create_rtree_heterogeneously_mergesort(struct rtree_point* points, const size_t len, const size_t threads);
struct rtree cuda_create_rtree_sisd(struct rtree_point* points, const size_t len);

void rtree_print_rect(struct rtree_rect r);
void rtree_print_node(struct rtree_node* n, const size_t depth);
void rtree_print(struct rtree tree);

#ifdef __cplusplus
}
#endif

#endif
