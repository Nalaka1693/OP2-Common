//
// auto-generated by op2.py
//

// user function
#include "../adt_calc.h"


// host stub function
void op_par_loop_adt_calc_execute(op_kernel_descriptor *desc) {

  op_set set = desc->set;
  char const *name = desc->name;
  int nargs = 6;

  op_arg arg0 = desc->args[0];
  op_arg arg1 = desc->args[1];
  op_arg arg2 = desc->args[2];
  op_arg arg3 = desc->args[3];
  op_arg arg4 = desc->args[4];
  op_arg arg5 = desc->args[5];

  op_arg args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};

// Compiling to Do JIT
#ifdef OP2_JIT
  if (!jit_compiled) {
    jit_compile();
  }
  (*adt_calc_function)(desc);
  return;
#endif

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timing_realloc(1);
  op_timers_core(&cpu_t1, &wall_t1);

  if (OP_diags > 2) {
    printf(" kernel routine with indirection: adt_calc\n");
  }

  int set_size = op_mpi_halo_exchanges(set, nargs, args);

  if (set->size > 0) {

    for (int n = 0; n < set_size; n++) {
      if (n == set->core_size) {
        op_mpi_wait_all(nargs, args);
      }
      int map0idx = arg0.map_data[n * arg0.map->dim + 0];
      int map1idx = arg0.map_data[n * arg0.map->dim + 1];
      int map2idx = arg0.map_data[n * arg0.map->dim + 2];
      int map3idx = arg0.map_data[n * arg0.map->dim + 3];

      adt_calc(&((double *)arg0.data)[2 * map0idx],
               &((double *)arg0.data)[2 * map1idx],
               &((double *)arg0.data)[2 * map2idx],
               &((double *)arg0.data)[2 * map3idx],
               &((double *)arg4.data)[4 * n], &((double *)arg5.data)[1 * n]);
    }
  }

  if (set_size == 0 || set_size == set->core_size) {
    op_mpi_wait_all(nargs, args);
  }
  // combine reduction data
  op_mpi_set_dirtybit(nargs, args);

  // update kernel record
  op_timers_core(&cpu_t2, &wall_t2);
  OP_kernels[1].name = name;
  OP_kernels[1].count += 1;
  OP_kernels[1].time += wall_t2 - wall_t1;
  OP_kernels[1].transfer += (float)set->size * arg0.size;
  OP_kernels[1].transfer += (float)set->size * arg4.size;
  OP_kernels[1].transfer += (float)set->size * arg5.size;
  OP_kernels[1].transfer += (float)set->size * arg0.map->dim * 4.0f;
}

void op_par_loop_adt_calc(char const *name, op_set set, op_arg arg0,
                          op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                          op_arg arg5) {

  op_kernel_descriptor *desc =
      (op_kernel_descriptor *)malloc(sizeof(op_kernel_descriptor));
  desc->name = name;
  desc->set = set;
  desc->device = 1;
  desc->index = 1;
  desc->hash = 5380;
  desc->hash = ((desc->hash << 5) + desc->hash) + 6;

  // save the iteration range

  // save the arguments
  desc->nargs = 6;
  desc->args = (op_arg *)malloc(6 * sizeof(op_arg));
  desc->args[0] = arg0;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg0.dat->index;
  desc->args[1] = arg1;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg1.dat->index;
  desc->args[2] = arg2;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg2.dat->index;
  desc->args[3] = arg3;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg3.dat->index;
  desc->args[4] = arg4;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg4.dat->index;
  desc->args[5] = arg5;
  desc->hash = ((desc->hash << 5) + desc->hash) + arg5.dat->index;
  desc->function = op_par_loop_adt_calc_execute;

  op_enqueue_kernel(desc);
}