#include "rtree.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <utility>

using std::vector;
using std::pair;
using std::make_pair;
using std::cout;
using std::endl;

#define title() do{printf("%s\n", __func__);} while(0)
/*
static inline void msg(const char* m, ...) {
  va_list args;
  va_start(args, m);
  vprintf(m, args);
}
*/

namespace
{
using std::chrono::duration;
using std::chrono::duration_cast;
typedef std::chrono::high_resolution_clock Clock;
}

const size_t PRINT_CUTOFF = 1000;

/// generate a uniform random between min and max exclusive
static inline ord_t uniform_frand(const ord_t min, const ord_t max) {
  const double r = (double)rand() / RAND_MAX;
  return min + r * (max - min);
}

static inline rtree_point* create_points_together(const size_t num) {
  const ord_t min = 0.0f;
  const ord_t max = 100.0f;

  rtree_point* points = (rtree_point*) malloc(sizeof(rtree_point) * num);
  for(size_t i = 0, end = num; i != end; ++i) {
    points[i].x = uniform_frand(min, max);
    points[i].y = uniform_frand(min, max);
    points[i].key = i;
  }
  return points;
}

static inline rtree_points create_points(const size_t num) {
  const ord_t min = 0.0f;
  const ord_t max = 100.0f;

  ord_t* x = (ord_t*) malloc(sizeof(ord_t) * num);
  rtree_y_key* ykey = (rtree_y_key*) malloc(sizeof(rtree_y_key) * num);
  rtree_points points = {x, ykey, num};
  for(size_t i = 0, end = num; i != end; ++i) {
    points.x[i] = uniform_frand(min, max);
    points.ykey[i].y = uniform_frand(min, max);
    points.ykey[i].key = i;
  }
  return points;
}

static inline void destroy_points(rtree_points points) {
  free(points.x);
  free(points.ykey);
}

static inline void print_points(rtree_points points) {
  printf("x\ty\tkey\n");
  for(size_t i = 0, end = points.length; i != end; ++i)
    printf("%f\t%f\t%d\n", points.x[i], points.ykey[i].y, points.ykey[i].key);
  printf("\n");
}

static inline void print_points_together(rtree_point* points, const size_t len) {
  printf("x\ty\tkey\n");
  for(size_t i = 0, end = len; i != end; ++i)
    printf("%f\t%f\t%d\n", points[i].x, points[i].y, points[i].key);
  printf("\n");
}


/// SIMD sort (CUDA)
static inline void test_rtree_simd(const size_t num) {
  title();

  rtree_points points = create_points(num);

  const auto start = Clock::now();
  rtree tree = cuda_create_rtree(points);
  const auto end = Clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "time (ms): " << elapsed_ms << endl;

  if(num < PRINT_CUTOFF) {
    print_points(points);
    rtree_print(tree);
  }

  destroy_points(points);
}

/// MIMD sort (multicore CPU via TBB)
static inline void test_rtree_mimd(const size_t num, const size_t threads) {
  title();

  rtree_point* points = create_points_together(num);

  const auto start = Clock::now();
  rtree tree = cuda_create_rtree_heterogeneously(points, num, threads);
  const auto end = Clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "time (ms): " << elapsed_ms << endl;

  if(num < PRINT_CUTOFF) {
    print_points_together(points, num);
    rtree_print(tree);
  }

  free(points);
}

/// SISD sort (single core CPU via std::sort)
static inline void test_rtree_sisd(const size_t num) {
  title();

  rtree_point* points = create_points_together(num);

  const auto start = Clock::now();
  rtree tree = cuda_create_rtree_sisd(points, num);
  const auto end = Clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  cout << "time (ms): " << elapsed_ms << endl;

  if(num < PRINT_CUTOFF) {
    print_points_together(points, num);
    rtree_print(tree);
  }

  free(points);
}


static inline void test_pipelined(const size_t len, const size_t threads) {
  title();
  const size_t PIPELINE_LEN = 10;

  printf("creating points...\n");
  rtree_point* points = create_points_together(len);
  printf("points: %lu\n", len);

  vector<pair<rtree_point*, size_t>> pointses;
  pointses.push_back(make_pair(points, len));
  for(size_t i = 0, end = PIPELINE_LEN; i != end; ++i) {
    rtree_point* morepoints = new rtree_point[len];
    memcpy(morepoints, points, len * sizeof(rtree_point));
    pointses.push_back(make_pair(morepoints, len));
  }

  cout << "creating tree..." << endl;

  const auto start = std::chrono::high_resolution_clock::now();

  vector<rtree> trees = rtree_create_pipelined(pointses, threads);

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "cpu time (ms): " << elapsed_ms << std::endl;
  printf("ms per point: %f\n", (double)elapsed_ms / len);

  for(int i = 0, end = PIPELINE_LEN; i != end; ++i)
    free(pointses[i].first);
}

static inline void test_unpipelined(const size_t len, const size_t threads) {
  const size_t PIPELINE_LEN = 10;
  printf("test_unpipelined\n");
  printf("creating points...\n");
  rtree_point* points = create_points_together(len);
  printf("points: %lu\n", len);

  vector<pair<rtree_point*, size_t>> pointses;
  pointses.push_back(make_pair(points, len));
  for(size_t i = 0, end = PIPELINE_LEN; i != end; ++i) {
    rtree_point* morepoints = new rtree_point[len];
    memcpy(morepoints, points, len * sizeof(rtree_point));
    pointses.push_back(make_pair(morepoints, len));
  }

  cout << "creating tree..." << endl;

  const auto start = std::chrono::high_resolution_clock::now();

  vector<rtree> trees;
  for(size_t i = 0, end = PIPELINE_LEN;i != end; ++i) {
    trees.push_back(cuda_create_rtree_heterogeneously(pointses[i].first, pointses[i].second, threads));
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "cpu time (ms): " << elapsed_ms << std::endl;
  printf("ms per point: %f\n", (double)elapsed_ms / len);

  for(int i = 0, end = PIPELINE_LEN; i != end; ++i)
    free(pointses[i].first);
}

struct app_arguments {
  bool        success;
  const char* app_name;
  size_t      test_num;
  size_t      array_size;
  size_t      threads;
};

static app_arguments parse_args(const int argc, const char** argv) {
  app_arguments args;
  args.success = false;

  int arg_i = 0;
  if(argc <= arg_i)
    return args;
  args.app_name = argv[arg_i];
  ++arg_i;

  if(argc <= arg_i)
    return args;
  args.test_num = strtol(argv[arg_i], NULL, 10);
  ++arg_i;

  if(argc <= arg_i)
    return args;
  args.array_size = strtol(argv[arg_i], NULL, 10);
  ++arg_i;

  if(args.test_num == 1 || args.test_num > 2) {
    if(argc <= arg_i)
      return args;
    args.threads = strtol(argv[arg_i], NULL, 10);
  }
  ++arg_i;

  args.success = true;
  return args;
}

static void print_usage(const char* app_name) {
  const char* default_app_name = "rtree";
  printf("usage: %s test_num array_size threads\n", strlen(app_name) == 0 ? default_app_name : app_name);
  printf("       test 0: CUDA construction\n");
  printf("       test 1: heterogeneous CUDA+MIMD construction\n");
  printf("       test 2: serial sort with CUDA construction\n");
  printf("       test 3: pipelined heterogeneous construction x10\n");
  printf("       test 4: unpipelined heterogeneous construction x10\n");
  printf(" *threads is ONLY used for test 1, and then only for the MIMD sort. 2xCores is generally good.\n");
  printf("\n");
}

int main(const int argc, const char** argv) {
  srand(4242); // seed deterministically, so results are reproducible

  const app_arguments args = parse_args(argc, argv);
  if(!args.success) {
    print_usage(args.app_name);
    return 0;
  }

  if(args.test_num == 0)
    test_rtree_simd(args.array_size);
  else if(args.test_num == 1)
    test_rtree_mimd(args.array_size, args.threads);
  else if(args.test_num == 2)
    test_rtree_sisd(args.array_size);
  else if(args.test_num == 3)
    test_pipelined(args.array_size, args.threads);
  else if(args.test_num == 4)
    test_unpipelined(args.array_size, args.threads);

  printf("\n");
  return 0;
}
