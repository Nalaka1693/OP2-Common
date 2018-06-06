/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __OP_CUDA_REDUCTION_SUPP_H
#define __OP_CUDA_REDUCTION_SUPP_H

/*
 * This file provides an implementation for calculation of supplementary data
 * for reduction of OP2 global variables and global constants.
 * It is separated from the op_cuda_rt_support.h file because the reduction code
 * is based on C++ templates, while the other file only includes C routines.
 */

/*
 * sturcture for reduction supplementary data
 */

typedef struct {
  int reduct_bytes; /* size of the reduction arrays in bytes */
  int reduct_size;  /* size of the biggest reduction type */
} reduct_supp_data_t;

/*
 * Reduction data calculating function
 */

__inline__ void op_setup_reductions(reduct_supp_data_t &t, op_arg *args,
                                    const int nargs, const int maxblocks) {
  t.reduct_bytes = 0;
  t.reduct_size = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ) {
      t.reduct_bytes += ROUND_UP(maxblocks * args[i].size);
      t.reduct_size = MAX(t.reduct_size, args[i].size / args[i].dim);
    }
  }
  reallocReductArrays(t.reduct_bytes);
  t.reduct_bytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ) {
      args[i].data = OP_reduct_h + t.reduct_bytes;
      args[i].data_d = OP_reduct_d + t.reduct_bytes;
      t.reduct_bytes += ROUND_UP(maxblocks * args[i].size);
    }
  }
}

/*
 * Initialisation of reduction arrays
 */

template <typename T, op_access reduction>
__inline__ void setRedArrToArg(op_arg &arg, const int maxblocks, T *argh) {
  for (int b = 0; b < maxblocks; b++) {
    for (int d = 0; d < arg.dim; d++) {
      if (reduction == OP_INC) {
        ((T *)arg.data)[d + b * arg.dim] = *argh;
      } else if (reduction == OP_MAX || reduction == OP_MIN) {
        ((T *)arg.data)[d + b * arg.dim] = argh[d];
      }
    }
  }
}

template <typename T, op_access reduction>
__inline__ void updateRedArrToArg(op_arg &arg, const int maxblocks, T *argh) {
  for (int b = 0; b < maxblocks; b++) {
    for (int d = 0; d < arg.dim; d++) {
      if (reduction == OP_INC) {
        argh[d] += ((T *)arg.data)[d + b * arg.dim];
      } else if (reduction == OP_MIN) {
        argh[d] = MIN(argh[d], ((T *)arg.data)[d + b * arg.dim]);
      } else if (reduction == OP_MAX) {
        argh[d] = MAX(argh[d], ((T *)arg.data)[d + b * arg.dim]);
      }
    }
  }
  arg.data = (char *)argh;
  op_mpi_reduce(&arg, argh);
}

/*
 * Calculation of the size of the const array needed
 */

__inline__ void op_setup_constants(int &const_bytes, op_arg *args,
                                    const int nargs) {
  const_bytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc == OP_READ) {
      const_bytes += ROUND_UP(args[i].size);
    }
  }
  reallocConstArrays(const_bytes);
  const_bytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc == OP_READ) {
      args[i].data = OP_consts_h + const_bytes;
      args[i].data_d = OP_consts_d + const_bytes;
      const_bytes += ROUND_UP(args[i].size);
    }
  }
}

/*
 * Initialisation of constant arrays -- std::copy 
 */

template <typename T>
__inline__ void setConstantArrToArg(op_arg &arg, T *argh) {
  for (int d = 0; d < arg.dim; d++) {
    ((T *)arg.data)[d] = argh[d];
  }
}

#endif /* __OP_CUDA_REDUCTION_SUPP_H */
