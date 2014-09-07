rtree-cuda
==========

CUDA R tree construction

## To Do
1. combine level kernel calls into single kernel
  * ```kernel() { while(more-levels) { c_create_level() } }```
2. use shared memory
3. add c construction back (from history), for comparison
