/* monolis_c_main.h */
#ifndef MONOLIS_C_MAIN_H
#define MONOLIS_C_MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "monolis_struct.h"

void monolis_com_input_comm_table(
  MONOLIS* mat);

void monolis_add_sparse_matrix(
  MONOLIS* mat,
  int      nbase_func,
  int      *connectivity,
  double** local_mat);

void monolis_copy_param(
  MONOLIS* in,
  MONOLIS* out);

void monolis_get_CRR_format(
  int      nnode,
  int      nz,
  int*     index,
  int*     item,
  int*     indexR,
  int*     itemR,
  int*     permR);

void monolis_add_sparse_matrix_c_main(
  int      nnode,
  int      nz,
  int      ndof,
  int      nbf,
  int*     index,
  int*     item,
  double*  A,
  int*     con,
  double*  mat);

void monolis_add_scalar_to_sparse_matrix_c_main(
  int      nnode,
  int      nz,
  int      ndof,
  int*     index,
  int*     item,
  double*  A,
  int      i,
  int      j,
  int      submat_i,
  int      submat_j,
  double   val);

void monolis_set_Dirichlet_bc_c_main(
  int      nnode,
  int      nz,
  int      ndof,
  int*     index,
  int*     item,
  int*     indexR,
  int*     itemR,
  int*     permR,
  double*  A,
  double*  b,
  int      node_id,
  int      ndof_bc,
  double   val);

void monolis_inner_product_c_main(
  int      n,
  int      ndof,
  double*  x,
  double*  y,
  double   sum,
  int      comm);

void monolis_solve_c_main(
  int      n,
  int      np,
  int      nz,
  int      ndof,
  double*  A,
  double*  x,
  double*  b,
  int*     index,
  int*     item,
  int      myrank,
  int      comm,
  int      commsize,
  int      recv_n_neib,
  int      recv_nitem,
  int*     recv_neib_pe,
  int*     recv_index,
  int*     recv_item,
  int      send_n_neib,
  int      send_nitem,
  int*     send_neib_pe,
  int*     send_index,
  int*     send_item,
  int      method,
  int      precond,
  int      maxiter,
  double   tol,
  int      iterlog,
  int      timelog,
  int      summary);

void monolis_qsort_int(
  int* array,
  int  iS,
  int  iE);

#ifdef __cplusplus
}
#endif

#endif
