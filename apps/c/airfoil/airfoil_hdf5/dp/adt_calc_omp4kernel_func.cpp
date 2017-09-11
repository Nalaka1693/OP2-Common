//
// auto-generated by op2.py
//

void adt_calc_omp4_kernel(
  int *map0,
  int map0size,
  double *data4,
  int dat4size,
  double *data5,
  int dat5size,
  double *data0,
  int dat0size,
  int *col_reord,
  int set_size1,
  int start,
  int end,
  int num_teams,
  int nthread,
  int opDat0_adt_calc_stride_OP2CONSTANT,
  int direct_adt_calc_stride_OP2CONSTANT){

  #pragma omp target teams distribute parallel for schedule(static,1)\
     num_teams(num_teams) thread_limit(nthread) map(to:data4[0:dat4size],data5[0:dat5size]) \
    map(to: gam_ompkernel, gm1_ompkernel, cfl_ompkernel)\
    map(to:col_reord[0:set_size1],map0[0:map0size],data0[0:dat0size])
  for ( int e=start; e<end; e++ ){
    int n_op = col_reord[e];
    int map0idx = map0[n_op + set_size1 * 0];
    int map1idx = map0[n_op + set_size1 * 1];
    int map2idx = map0[n_op + set_size1 * 2];
    int map3idx = map0[n_op + set_size1 * 3];

    //variable mapping
    const double *x1 = &data0[map0idx];
    const double *x2 = &data0[map1idx];
    const double *x3 = &data0[map2idx];
    const double *x4 = &data0[map3idx];
    const double *q = &data4[n_op];
    double *adt = &data5[1*n_op];

    //inline function
      
    double dx, dy, ri, u, v, c;
  
    ri = 1.0f / q[(0)*direct_adt_calc_stride_OP2CONSTANT];
    u = ri * q[(1)*direct_adt_calc_stride_OP2CONSTANT];
    v = ri * q[(2)*direct_adt_calc_stride_OP2CONSTANT];
    c = sqrt(gam_ompkernel * gm1_ompkernel * (ri * q[(3)*direct_adt_calc_stride_OP2CONSTANT] - 0.5f * (u * u + v * v)));
  
    dx = x2[(0)*opDat0_adt_calc_stride_OP2CONSTANT] - x1[(0)*opDat0_adt_calc_stride_OP2CONSTANT];
    dy = x2[(1)*opDat0_adt_calc_stride_OP2CONSTANT] - x1[(1)*opDat0_adt_calc_stride_OP2CONSTANT];
    *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);
  
    dx = x3[(0)*opDat0_adt_calc_stride_OP2CONSTANT] - x2[(0)*opDat0_adt_calc_stride_OP2CONSTANT];
    dy = x3[(1)*opDat0_adt_calc_stride_OP2CONSTANT] - x2[(1)*opDat0_adt_calc_stride_OP2CONSTANT];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);
  
    dx = x4[(0)*opDat0_adt_calc_stride_OP2CONSTANT] - x3[(0)*opDat0_adt_calc_stride_OP2CONSTANT];
    dy = x4[(1)*opDat0_adt_calc_stride_OP2CONSTANT] - x3[(1)*opDat0_adt_calc_stride_OP2CONSTANT];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);
  
    dx = x1[(0)*opDat0_adt_calc_stride_OP2CONSTANT] - x4[(0)*opDat0_adt_calc_stride_OP2CONSTANT];
    dy = x1[(1)*opDat0_adt_calc_stride_OP2CONSTANT] - x4[(1)*opDat0_adt_calc_stride_OP2CONSTANT];
    *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);
  
    *adt = (*adt) * (1.0f / cfl_ompkernel);
    //end inline func
  }

}
