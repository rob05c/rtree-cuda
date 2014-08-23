#include "rtree.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// ANSI color codes
#define KNRM "\x1b[0m"
#define KBLK "\x1b[30m"
#define KRED "\x1b[31m"
#define KGRN "\x1b[32m"
#define KYEL "\x1b[33m"
#define KBLU "\x1b[34m"
#define KMAG "\x1b[35m"
#define KCYN "\x1b[36m"
#define KGRY "\x1b[1;30m"
#define KBGRY "\x1b[37m"
#define KBBLK "\x1b[1;30m"
#define KBRED "\x1b[1;31m"
#define KBGRN "\x1b[1;32m"
#define KBYEL "\x1b[1;33m"
#define KBBLU "\x1b[1;34m"
#define KBMAG "\x1b[1;35m"
#define KBCYN "\x1b[1;36m"
#define KWHT "\x1b[1;37m"
#define KRESET "\033[0m"

// These short names are really only ok because main only has tests in it.
// If other stuff is added to main, the testing stuff ought to be put in its own file.
#define KTITLE KWHT
#define KMSG KBGRY
#define KTEST_MSG KBGRY // \todo remove
#define title() do{printf(KTITLE "%s" KRESET "\n", __func__);} while(0)
static inline void msg(const char* m, ...) {
  va_list args;
  va_start(args, m);
  printf(KMSG);
  vprintf(m, args);
  printf(KRESET);
}

// generate a uniform random between min and max exclusive
static inline ord_t uniformFrand(const ord_t min, const ord_t max) {
  const double r = (double)rand() / RAND_MAX;
  return min + r * (max - min);
}

static inline void test_endian_2(const size_t len) {
  title();

//  static_assert(sizeof(unsigned int) == 4, "sizeof(int) is not 4, fix the below code")

  unsigned char a[4];
  unsigned char* array = a;
  // array[0] = 11
  array[0] = 0x0;
  array[0] = (array[0] << 2) | 0x1;
  array[0] = (array[0] << 2) | 0x2;
  array[0] = (array[0] << 2) | 0x3;

  // array[1] = 10
  array[1] = 0x3;
  array[1] = (array[1] << 2) | 0x2;
  array[1] = (array[1] << 2) | 0x1;
  array[1] = (array[1] << 2) | 0x0;

  // array[2] = 01
  array[2] = 0x0;
  array[2] = (array[2] << 2) | 0x3;
  array[2] = (array[2] << 2) | 0x2;
  array[2] = (array[2] << 2) | 0x1;

  // array[3] = 00 00 00 00
  array[3] = 0x3;
  array[3] = (array[3] << 2) | 0x2;
  array[3] = (array[3] << 2) | 0x0;
  array[3] = (array[3] << 2) | 0x1;

    unsigned int* iarray = (unsigned int*)array;
//  unsigned int endian = (array[0] << 24) | (array[1] << 16) | (array[2] << 8) | array[3];
  
    msg("endian: %u\n", *iarray);
}

static inline void test_many(const size_t len) {
  title();

  struct lqt_point* points = malloc(len * sizeof(struct lqt_point));
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = len; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  {
    msg("creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify(points, len, 
                                                     min, max, min, max, &depth);
    msg("sorting...\n");
    lqt_sortify(lqt);
    msg("\ndone\n");
    lqt_print_nodes(lqt, false);
    lqt_delete(lqt);
  }

  {
    msg("cuda creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify_cuda(points, len, min, max, min, max, &depth);
    msg("cuda sorting...\n");
    lqt_sortify_cuda(lqt);
    msg("\ncuda done\n");
    lqt_print_nodes(lqt, false);
    lqt_delete(lqt);
  }

}

static inline void test_endian(const size_t len) {
  title();

  typedef unsigned char uchar;
  typedef unsigned long sort_t;

  //  const unsigned short esa8[8] = {7, 6, 5, 4, 3, 2, 1, 0}; ///< lookup table
  //# define ENDIANSWAP8(a) (esa8[(a) % 8] + (a) / 8 * 8)

  //  const unsigned short esa4[4] = {3, 2, 1, 0}; ///< lookup table
  //# define ENDIANSWAP4(a) (esa4[(a) % 4] + (a) / 4 * 4)

  uchar chars[8];
  chars[0] = 37;
  chars[1] = 228;
  chars[2] = 99;
  chars[3] = 42;

//  sort_t* ichars = (sort_t*)chars; 
//  sort_t val = *ichars;
//  std::cout << "val " << val << std::endl;

//  uchar newchars[4];
//  for(int i = 0, end = sizeof(sort_t); i != end; ++i)
//    newchars[i] = chars[ENDIANSWAP4(i)];
//
//  ichars = (sort_t*)newchars; 
//  val = *ichars;

  sort_t val = 0;
  val = chars[3] | (chars[2] << 8) | (chars[1] << 16) | (chars[0] << 24);
  
  msg("eval %lu\n", val);
}

static inline void test_few(const size_t len) {
  title();

  struct lqt_point* points = malloc(len * sizeof(struct lqt_point));
  const ord_t min = 0.0;
  const ord_t max = 300.0;
  msg("creating points...\n");
  points[0].x = 299.999;
  points[0].y = 299.999;
  points[0].key = 42;
  points[1].x = 7.0;
  points[1].y = 14.0;
  points[1].key = 99;

  {
    msg("creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify(points, len, 
                                        min, max, min, max, &depth);
    msg("sorting...\n");
    lqt_sortify(lqt);
    msg("\ndone\n");
    lqt_print_nodes(lqt, true);
    lqt_delete(lqt);
  }

  {
    msg("cuda creating nodes...\n");
    size_t depth;
    struct linear_quadtree lqt = lqt_nodify_cuda(points, len, 
                                             min, max, min, max, &depth);
    msg("cuda sorting...\n");
    lqt_sortify_cuda(lqt);
    msg("\ncuda done\n");
    lqt_print_nodes(lqt, true);
  }

}

static inline void test_time(const size_t numPoints) {
  title();
  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  size_t depth;
  msg("cpu nodify...\n");
  const clock_t start = clock();
  struct linear_quadtree lqt = lqt_nodify(points, numPoints, 
                                      min, max, min, max, &depth);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  msg("cpu nodify time: %fs\n", elapsed_s);
  lqt_delete(lqt);
  // lqt and points not valid henceforth and hereafter.

  msg("creating cuda points...\n");
  struct lqt_point* cuda_points = malloc(sizeof(struct lqt_point) * numPoints);
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    cuda_points[i].x = uniformFrand(min, max);
    cuda_points[i].y = uniformFrand(min, max);
    cuda_points[i].key = i;
  }

  msg("gpu nodify...\n");
  const clock_t start_cuda = clock();
  struct linear_quadtree cuda_lqt = lqt_nodify_cuda(cuda_points, numPoints, 
                                                min, max, min, max, &depth);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double speedup = elapsed_s / elapsed_s_cuda;
  msg("gpu nodify time: %fs\n", elapsed_s_cuda);
  msg("gpu speedup: %f\n", speedup);
  lqt_delete(cuda_lqt);
}

static inline void test_sorts(const size_t numPoints) {
  title();

  struct lqt_point* points = malloc(numPoints * sizeof(struct lqt_point));
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  msg("creating nodes...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_nodify(points, numPoints, 
                                            min, max, min, max, &depth);
  struct linear_quadtree qt_cuda;
  lqt_copy(&qt_cuda, &qt);

  msg("sorting...\n");
  lqt_sortify(qt);
  msg("sorting cuda...\n");
  lqt_sortify_cuda(qt_cuda);

  msg("nodes:\n");
  lqt_print_nodes(qt, false);
  msg("cuda nodes:\n");
  lqt_print_nodes(qt_cuda, false);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_sort_time(const size_t numPoints) {
  title();

  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }

  msg("creating nodes...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_nodify(points, numPoints, 
                                     min, max, min, max, &depth);
  struct linear_quadtree qt_cuda;
  lqt_copy(&qt_cuda, &qt);

  msg("sorting...\n");
  const clock_t start = clock();
  lqt_sortify(qt);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  msg("sort time: %fs\n", elapsed_s);

  msg("sorting cuda...\n");
  const clock_t start_cuda = clock();
  lqt_sortify_cuda(qt_cuda);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double cuda_speedup = elapsed_s / elapsed_s_cuda;
  msg("cuda sort time: %fs\n", elapsed_s_cuda);
  msg("cuda speedup: %f\n", cuda_speedup);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_unified_sorts(const size_t numPoints) {
  title();

  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }
  struct lqt_point* points_cuda = malloc(numPoints * sizeof(struct lqt_point));
  memcpy(points_cuda, points, numPoints * sizeof(struct lqt_point));

  msg("points: %lu\n", numPoints);

  msg("creating quadtree...\n");
  size_t depth;
  struct linear_quadtree qt = lqt_create(points, numPoints, 
                                         min, max, min, max, &depth);
  msg("creating quadtree with CUDA...\n");
  struct linear_quadtree qt_cuda = lqt_create_cuda(points_cuda, numPoints, 
                                                   min, max, min, max, &depth);
  msg("nodes:\n");
  lqt_print_nodes(qt, false);
  msg("cuda nodes:\n");
  lqt_print_nodes(qt_cuda, false);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

static inline void test_unified(const size_t numPoints) {
  title();

  struct lqt_point* points = malloc(sizeof(struct lqt_point) * numPoints);
  const size_t min = 1000;
  const size_t max = 1100;
  msg("creating points...\n");
  for(int i = 0, end = numPoints; i != end; ++i) {
    points[i].x = uniformFrand(min, max);
    points[i].y = uniformFrand(min, max);
    points[i].key = i;
  }
  struct lqt_point* points_cuda = malloc(numPoints * sizeof(struct lqt_point));
  memcpy(points_cuda, points, numPoints * sizeof(struct lqt_point));

  msg("points: %lu\n", numPoints);
  msg("creating quadtree...\n");
  const clock_t start = clock();
  size_t depth;
  struct linear_quadtree qt = lqt_create(points, numPoints, 
                                         min, max, min, max, &depth);
  const clock_t end = clock();
  const double elapsed_s = (end - start) / (double)CLOCKS_PER_SEC;
  msg("cpu time: %fs\n", elapsed_s);
  msg("ms per point: %f\n", 1000.0 * elapsed_s / numPoints);

  msg("creating quadtree with CUDA...\n");
  const clock_t start_cuda = clock();
  struct linear_quadtree qt_cuda = lqt_create_cuda(points_cuda, numPoints, 
                                                   min, max, min, max, &depth);
  const clock_t end_cuda = clock();
  const double elapsed_s_cuda = (end_cuda - start_cuda) / (double)CLOCKS_PER_SEC;
  const double cuda_speedup = elapsed_s / elapsed_s_cuda;
  msg("cuda time: %fs\n", elapsed_s_cuda);
  msg("ms per cuda point: %f\n", 1000.0 * elapsed_s_cuda / numPoints);
  msg("cuda speedup: %f\n", cuda_speedup);

  lqt_delete(qt);
  lqt_delete(qt_cuda);
}

void(*test_funcs[])(const size_t) = {
  test_endian_2,
  test_many,
  test_endian,
  test_few,
  test_time,
  test_sorts,
  test_sort_time,
  test_unified,
  test_unified_sorts,
};

static const char* default_app_name = "mergesort";

const char* tests[][2] = {
  {"test_endian_2"     , "test endianness conversions between 4-byte array"},
  {"test_many"         , "print brief reports for many points"},
  {"test_endian"       , "test endian shifting in 4-byte array"},
  {"test_few"          , "print detailed reports for a few points"},
  {"test_time"         , "benchmark the time to create nodes using CPU vs CUDA"},
  {"test_sorts"        , "test the values produced by sorting with CPU vs CUDA"},
  {"test_sort_time"    , "benchmark the time to sort using CPU vs CUDA"},
  {"test_unified"      , "benchmark the time to create and sort using CPU vs CUDA"},
  {"test_unified_sorts", "test the values produced by CPU vs CUDA with unified create+sort function"},
};

const size_t test_num = sizeof(tests) / (sizeof(const char*) * 2);

struct app_arguments {
  bool        success;
  const char* app_name;
  size_t      test_num;
  size_t      array_size;
};

static struct app_arguments parseArgs(const int argc, const char** argv) {
  struct app_arguments args;
  args.success = false;

  if(argc < 1)
    return args;
  args.app_name = argv[0];

  if(argc < 2)
    return args;
  args.test_num = strtol(argv[1], NULL, 10);

  if(argc < 3)
    return args;
  args.array_size = strtol(argv[2], NULL, 10);

  args.success = true;
  return args;
}

/// \param[out] msg
/// \param[out] msg_len
static void print_usage(const char* app_name) {
  printf("usage: %s test_num  array_size\n", strlen(app_name) == 0 ? default_app_name : app_name);
  printf("\n");
  printf(KMAG "       num" KRESET KCYN " test" KRESET KGRN "            description\n" KRESET);
  for(size_t i = 0, end = test_num; i != end; ++i) {
    printf(KBMAG "       %-3.1lu" KRESET KBCYN " %-15.15s" KRESET KBGRN " %s\n" KRESET, i, tests[i][0], tests[i][1]);
  }
  printf("\n");
}

int main(const int argc, const char** argv) {
  srand(time(NULL));

  const struct app_arguments args = parseArgs(argc, argv);
  if(!args.success) {
    print_usage(args.app_name);
    return 0;
  }

  test_funcs[args.test_num](args.array_size);
  printf("\n");
  return 0;
}
