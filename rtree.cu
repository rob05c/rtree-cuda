#define CUB_STDERR

#include "rtree.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include <assert.h>
#include <math.h>
#include <linux/cuda.h>
#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "tbb/tbb.h"

using namespace cub; // debug

/// \todo fix to not be global
CachingDeviceAllocator g_allocator(true); // CUB caching allocator for device memory

inline void update_boundary(struct rtree_rect* boundary, struct rtree_point* p) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fmin(p->y, boundary->top);
  boundary->bottom = fmax(p->y, boundary->bottom);
  boundary->left = fmin(p->x, boundary->left);
  boundary->right = fmax(p->x, boundary->right);
}

__device__ void c_update_boundary(struct rtree_rect* boundary, const struct rtree_point* p) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fminf(p->y, boundary->top);
  boundary->bottom = fmaxf(p->y, boundary->bottom);
  boundary->left = fminf(p->x, boundary->left);
  boundary->right = fmaxf(p->x, boundary->right);
}

inline void update_boundary(struct rtree_rect* boundary, struct rtree_rect* node) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fmin(node->top, boundary->top);
  boundary->bottom = fmax(node->bottom, boundary->bottom);
  boundary->left = fmin(node->left, boundary->left);
  boundary->right = fmax(node->right, boundary->right);
}

__device__ void c_update_boundary(rtree_rect* boundary, rtree_rect* node) {
  /// \todo replace these with CUDA min/max which won't use conditionals
  boundary->top = fminf(node->top, boundary->top);
  boundary->bottom = fmaxf(node->bottom, boundary->bottom);
  boundary->left = fminf(node->left, boundary->left);
  boundary->right = fmaxf(node->right, boundary->right);
}

/// initialize boundary so the first udpate overrides it.
inline void init_boundary(struct rtree_rect* boundary) {
  boundary->top = ord_t_max;
  boundary->bottom = ord_t_lowest;
  boundary->left = ord_t_max;
  boundary->right = ord_t_lowest;
}

/// initialize boundary so the first udpate overrides it.
__device__ void c_init_boundary(rtree_rect* boundary) {
  boundary->top = ord_t_max;
  boundary->bottom = ord_t_lowest;
  boundary->left = ord_t_max;
  boundary->right = ord_t_lowest;
}


/// used to calculate tree height
/// \todo use CUDA maths
inline size_t log_base_ceil(const size_t x, const size_t base) {
  return (size_t)ceil(log((double)x) / log((double)base));
}


inline size_t get_node_length(const size_t i, const size_t level_len, const size_t previous_level_len, const size_t node_size) {
  // let would be nice.
  const size_t n = node_size;
  const size_t len = previous_level_len;
  const size_t final_i = level_len - 1; // this better be optimised out
  // this nasty bit sets lnum to len % n if it's the last iteration and there's a remainder, else n
  // which avoids a GPU-breaking conditional
  return ((i != final_i || len % n == 0) * n) + ((i == final_i && len % n != 0) * (len % n));
}

__device__ size_t c_get_node_length(const size_t i, const size_t level_len, const size_t previous_level_len, const size_t node_size) {
  // let would be nice.
  const size_t n = node_size;
  const size_t len = previous_level_len;
  const size_t final_i = level_len - 1; // this better be optimised out
  // this nasty bit sets lnum to len % n if it's the last iteration and there's a remainder, else n
  // which avoids a GPU-breaking conditional
  return ((i != final_i || len % n == 0) * n) + ((i == final_i && len % n != 0) * (len % n));
}

struct rtree cuda_create_rtree_heterogeneously(struct rtree_point* points, const size_t len) {
  struct rtree_leaf* leaves = cuda_create_leaves_together(tbb_sort(points, len), len);
  const size_t leaves_len = DIV_CEIL(len, RTREE_NODE_SIZE);

  rtree_node* previous_level = (rtree_node*) leaves;
  size_t      previous_len = leaves_len;
  size_t      depth = 1; // leaf level is 0
  while(previous_len > RTREE_NODE_SIZE) {
    previous_level = cuda_create_level(previous_level, previous_len);
    previous_len = DIV_CEIL(previous_len, RTREE_NODE_SIZE);
    ++depth;
  }

  rtree_node* root = (rtree_node*) malloc(sizeof(rtree_node));
  init_boundary(&root->bounding_box);
  root->num = previous_len;
  root->children = previous_level;
  for(size_t i = 0, end = previous_len; i != end; ++i)
    update_boundary(&root->bounding_box, &root->children[i].bounding_box);
  ++depth;

  struct rtree tree = {depth, root};
  return tree;
}

struct rtree cuda_create_rtree(struct rtree_points points) {
  struct rtree_leaf* leaves = cuda_create_leaves(cuda_sort(points));
  const size_t leaves_len = DIV_CEIL(points.length, RTREE_NODE_SIZE);

  rtree_node* previous_level = (rtree_node*) leaves;
  size_t      previous_len = leaves_len;
  size_t      depth = 1; // leaf level is 0
  while(previous_len > RTREE_NODE_SIZE) {
    previous_level = cuda_create_level(previous_level, previous_len);
    previous_len = DIV_CEIL(previous_len, RTREE_NODE_SIZE);
    ++depth;
  }

  rtree_node* root = (rtree_node*) malloc(sizeof(rtree_node));
  init_boundary(&root->bounding_box);
  root->num = previous_len;
  root->children = previous_level;
  for(size_t i = 0, end = previous_len; i != end; ++i)
    update_boundary(&root->bounding_box, &root->children[i].bounding_box);
  ++depth;

  struct rtree tree = {depth, root};
  return tree;
}

/// \param real_nodes NOT CUDA MEMORY! CANNOT BE ACCESSED!
__global__ void create_level_kernel(rtree_node* next_level, rtree_node* nodes, rtree_node* real_nodes, const size_t len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t next_level_len = DIV_CEIL(len, RTREE_NODE_SIZE);

  if(i >= next_level_len)
    return; // skip the final block remainder

  rtree_node* n = &next_level[i];
  c_init_boundary(&n->bounding_box);
  n->num = c_get_node_length(i, next_level_len, len, RTREE_NODE_SIZE);
  n->children = &real_nodes[i * RTREE_NODE_SIZE];

# pragma unroll
  for(size_t j = 0, jend = n->num; j != jend; ++j)
    c_update_boundary(&n->bounding_box, &nodes[i * RTREE_NODE_SIZE + j].bounding_box);
}

/// \param nodes Can really be either a rtree_node or rtree_leaf; doesn't matter to us, we won't dereference .children
/// \return next level up. Length is ceil(nodes_len / RTREE_NODE_SIZE)
struct rtree_node* cuda_create_level(struct rtree_node* nodes, const size_t nodes_len) {
  const size_t THREADS_PER_BLOCK = 512;
  const size_t next_level_len = DIV_CEIL(nodes_len, RTREE_NODE_SIZE);

//  init_boundary(&next_level->bounding_box);

  rtree_node* cuda_nodes;
  rtree_node* cuda_next_level;
  cudaMalloc((void**)&cuda_nodes, nodes_len * sizeof(rtree_node));
  cudaMalloc((void**)&cuda_next_level, next_level_len * sizeof(rtree_node));

  cudaMemcpy(cuda_nodes, nodes, nodes_len * sizeof(rtree_node), cudaMemcpyHostToDevice);

  create_level_kernel<<<(next_level_len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_next_level, cuda_nodes, nodes, nodes_len);

  rtree_node* next_level = (rtree_node*) malloc(sizeof(rtree_node) * next_level_len);
  cudaMemcpy(next_level, cuda_next_level, next_level_len * sizeof(rtree_node), cudaMemcpyDeviceToHost);

  cudaFree(cuda_next_level);
  cudaFree(cuda_nodes);

  return next_level;
}

/// \param real_points NOT CUDA MEMORY! CANNOT BE ACCESSED!
__global__ void create_leaves_together_kernel(rtree_leaf* leaves, rtree_point* points, rtree_point* real_points, const size_t len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t len_leaves = DIV_CEIL(len, RTREE_NODE_SIZE);

  if(i >= len_leaves)
    return; // skip the final block remainder

  rtree_leaf* l = &leaves[i];
  c_init_boundary(&l->bounding_box);
  l->num = c_get_node_length(i, len_leaves, len, RTREE_NODE_SIZE);
  l->points = &real_points[i * RTREE_NODE_SIZE];

# pragma unroll
  for(size_t j = 0, jend = l->num; j != jend; ++j) {
    const rtree_point* p = &points[i * RTREE_NODE_SIZE + j];
    c_update_boundary(&l->bounding_box, p);
  }
}

struct rtree_leaf* cuda_create_leaves_together(struct rtree_point* sorted, const size_t len) {
  static_assert(sizeof(rtree_node) == sizeof(rtree_leaf), "rtree node, leaf sizes must be equal, since leaves are passed to create_level");

  const size_t THREADS_PER_BLOCK = 512;

  const size_t leaves_num = DIV_CEIL(len, RTREE_NODE_SIZE);

  rtree_leaf*  cuda_leaves;
  rtree_point* cuda_points;

  cudaMalloc((void**)&cuda_leaves, leaves_num * sizeof(rtree_leaf));
  cudaMalloc((void**)&cuda_points, len * sizeof(rtree_point));

  cudaMemcpy(cuda_points, sorted, len * sizeof(rtree_point), cudaMemcpyHostToDevice);

  create_leaves_together_kernel<<<(leaves_num + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_leaves, cuda_points, sorted, len);

  rtree_leaf* leaves = (rtree_leaf*) malloc(sizeof(rtree_leaf) * leaves_num);

  cudaMemcpy(leaves, cuda_leaves, leaves_num * sizeof(rtree_leaf), cudaMemcpyDeviceToHost);

  cudaFree(cuda_leaves);
  cudaFree(cuda_points);

  return leaves;
}


/// \param real_points NOT CUDA MEMORY! CANNOT BE ACCESSED!
__global__ void create_leaves_kernel(rtree_leaf* leaves, rtree_point* points, rtree_point* real_points, ord_t* x, rtree_y_key* ykey, const size_t len) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  const size_t len_leaves = DIV_CEIL(len, RTREE_NODE_SIZE);

  if(i >= len_leaves)
    return; // skip the final block remainder

  rtree_leaf* l = &leaves[i];
  c_init_boundary(&l->bounding_box);
  l->num = c_get_node_length(i, len_leaves, len, RTREE_NODE_SIZE);
  l->points = &real_points[i * RTREE_NODE_SIZE];

# pragma unroll
  for(size_t j = 0, jend = l->num; j != jend; ++j) {
    rtree_point* p = &points[i * RTREE_NODE_SIZE + j];
    p->x   = x[i * RTREE_NODE_SIZE + j];
    p->y   = ykey[i * RTREE_NODE_SIZE + j].y;
    p->key = ykey[i * RTREE_NODE_SIZE + j].key;
    c_update_boundary(&l->bounding_box, p);
  }
}

/// \todo make CUDA
struct rtree_leaf* cuda_create_leaves(struct rtree_points sorted) {
  static_assert(sizeof(rtree_node) == sizeof(rtree_leaf), "rtree node, leaf sizes must be equal, since leaves are passed to create_level");

  const size_t THREADS_PER_BLOCK = 512;

  const size_t len = sorted.length;
  const size_t leaves_num = DIV_CEIL(sorted.length, RTREE_NODE_SIZE);


  rtree_leaf*  cuda_leaves;
  rtree_point* cuda_points;
  ord_t*       cuda_x;
  rtree_y_key* cuda_ykey;

  cudaMalloc((void**)&cuda_leaves, leaves_num    * sizeof(rtree_leaf));
  cudaMalloc((void**)&cuda_points, sorted.length * sizeof(rtree_point));
  cudaMalloc((void**)&cuda_x,      sorted.length * sizeof(ord_t));
  cudaMalloc((void**)&cuda_ykey,   sorted.length * sizeof(rtree_y_key));

  cudaMemcpy(cuda_x,    sorted.x,    sorted.length * sizeof(ord_t),       cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_ykey, sorted.ykey, sorted.length * sizeof(struct rtree_y_key), cudaMemcpyHostToDevice);

  rtree_point* points = (rtree_point*) malloc(sizeof(rtree_point) * sorted.length);

  create_leaves_kernel<<<(leaves_num + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(cuda_leaves, cuda_points, points, cuda_x, cuda_ykey, len);

  rtree_leaf*  leaves = (rtree_leaf*)  malloc(sizeof(rtree_leaf) * leaves_num);


  cudaMemcpy(leaves, cuda_leaves, leaves_num    * sizeof(rtree_leaf), cudaMemcpyDeviceToHost);
  cudaMemcpy(points, cuda_points, sorted.length * sizeof(rtree_point), cudaMemcpyDeviceToHost);

//  for(size_t i = 0, end = leaves_num; i != end; ++i)
//    leaves[i].points = &points[i * RTREE_NODE_SIZE];

  cudaFree(cuda_x);
  cudaFree(cuda_ykey);
  cudaFree(cuda_leaves);
  cudaFree(cuda_points);

  return leaves;
}

// x value ALONE is used for comparison, to create an xpack
bool operator<(const rtree_point& rhs, const rtree_point& lhs) {
  return rhs.x < lhs.x;
}

struct rtree_point* tbb_sort(struct rtree_point* points, const size_t len) {
//  auto lowxpack = [](const struct rtree_point& rhs, const struct rtree_point& lhs) {
//    return rhs.x < rhs.y;
//  };
  tbb::parallel_sort(points, points + len);
  return points;
}

struct rtree_points cuda_sort(struct rtree_points points) {
  typedef ord_t key_t;
  typedef struct rtree_y_key value_t;
  DoubleBuffer<key_t> d_keys;
  DoubleBuffer<value_t> d_values;
  CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(key_t) * points.length));
  CubDebugExit( g_allocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(key_t) * points.length));
  CubDebugExit( g_allocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(value_t) * points.length));
  CubDebugExit( g_allocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(value_t) * points.length));

  CubDebugExit( cudaMemcpy(d_keys.d_buffers[0], points.x, sizeof(key_t) * points.length, cudaMemcpyHostToDevice));
  CubDebugExit( cudaMemcpy(d_values.d_buffers[0], points.ykey, sizeof(value_t) * points.length, cudaMemcpyHostToDevice));

  size_t temp_storage_bytes = 0;
  void* d_temp_storage = NULL;
  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, points.length));
  CubDebugExit( g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  CubDebugExit( DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, points.length));
  
  CubDebugExit( cudaMemcpy(points.x, d_keys.Current(), points.length * sizeof(key_t), cudaMemcpyDeviceToHost));
  CubDebugExit( cudaMemcpy(points.ykey, d_values.Current(), points.length * sizeof(value_t), cudaMemcpyDeviceToHost));

  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_keys.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[0]));
  CubDebugExit( g_allocator.DeviceFree(d_values.d_buffers[1]));
  CubDebugExit( g_allocator.DeviceFree(d_temp_storage));

  return points;
}
