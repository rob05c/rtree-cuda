#include "rtree.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

inline void rtree_print_point(const struct rtree_point* point) {
  printf("[%f,%f,%d]", point->x, point->y, point->key);
}

inline void rtree_print_rect(struct rtree_rect r) {
  printf("[%f,%f,%f,%f]", r.top, r.left, r.bottom, r.right);
}

void rtree_print_leaf(const struct rtree_leaf* leaf) {
  printf("{\"bounding-box\":");
  rtree_print_rect(leaf->bounding_box);
  printf(", \"points\":[");

  for(size_t i = 0, end = leaf->num; i != end; ++i) {
    rtree_print_point(&leaf->points[i]);
    if(i + 1 != end)
      printf(",");
  }
  printf("]}");
}

void rtree_print_leaves(const struct rtree_leaf* leaves, const size_t len) {
  printf("[\n");
  for(size_t i = 0, end = len; i != end; ++i) {
    rtree_print_leaf(&leaves[i]);
    if(i + 1 != end)
      printf(",\n");
  }
  printf("\n]\n");
}

void rtree_print_node(struct rtree_node* n, size_t depth) {
  if(depth == 0) {
    rtree_print_leaf((struct rtree_leaf*)n);
    return;
  }

  printf("{\"bounding-box\":");
  rtree_print_rect(n->bounding_box);
  printf(", \"children\":[");
  printf("\n");

  for(size_t i = 0, end = n->num; i != end; ++i) {
    rtree_print_node(&n->children[i], depth - 1);
    if(i + 1 != end) {
      printf(",");
      printf("\n");
    }
  }

  printf("]}");
}

void rtree_print(struct rtree tree) {
  rtree_print_node(tree.root, tree.depth - 1);
  printf("\n");
}
