#ifndef rtreeH
#define rtreeH
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <limits.h>
#include <float.h>
#include <vector>
#include <utility>

// efficient ceil((float)a/(float)b)
#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

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
  rtree_y_key* ykey;
  size_t length;
};


#define RTREE_NODE_SIZE 4

struct rtree_node;
struct rtree {
  size_t depth; // needed to know where the leaves begin
  rtree_node* root;
};

struct rtree_rect {
  ord_t top;
  ord_t bottom;
  ord_t left;
  ord_t right;
};

struct rtree_leaf {
  rtree_rect   bounding_box; ///< MUST be first, so leaf boundary can be checked as node
  size_t       num;
  rtree_point* points;
};

struct rtree_node {
  rtree_rect  bounding_box; ///< MUST be first, so leaf boundary can be checked as node
  size_t      num;
  rtree_node* children;
};

void rtree_print_leaves(const rtree_leaf* leaves, const size_t len);
void rtree_print_leaf(const rtree_leaf* leaf);
void rtree_print_point(const rtree_point* point);

rtree_points cuda_sort(rtree_points points);
rtree_leaf* cuda_create_leaves(rtree_points sorted);
rtree_node* cuda_create_level(rtree_node* nodes, const size_t nodes_len);
rtree cuda_create_rtree(rtree_points points);

rtree cuda_create_rtree_heterogeneously(rtree_point* points, const size_t len, const size_t threads);
rtree_point* tbb_sort(rtree_point* points, const size_t len, const size_t threads);
rtree_leaf* cuda_create_leaves_together(rtree_point* sorted, const size_t len);

rtree cuda_create_rtree_heterogeneously_mergesort(rtree_point* points, const size_t len, const size_t threads);
rtree cuda_create_rtree_sisd(rtree_point* points, const size_t len);

std::vector<rtree> rtree_create_pipelined(std::vector< std::pair<rtree_point*, size_t> > pointses, const size_t threads);

void rtree_print_rect(rtree_rect r);
void rtree_print_node(rtree_node* n, const size_t depth);
void rtree_print(rtree tree);

#endif
