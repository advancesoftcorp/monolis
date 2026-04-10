#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>

#include "mpi.h"

#include "monolis.h"
#include "monolis_mpi_c.h"

void monolis_BCSR_solve_c_test_R(){
  MONOLIS mat;
  MONOLIS_COM com;
  int n_vertex;
  int n_node;
  int n_dof;
  int i, j, k;
  int comm_size;
  int my_rank;
  double val;
  double a[20];
  double b[20];
  double local_dense_matrix[20][20];
  int global_index[20];
  int dof_list[20];

  n_vertex = 20;

  monolis_std_log_string("monolis_BCSR_solve_c_test_R");

  n_node = 10;
  n_dof  = 2;

  monolis_initialize(&mat);
  comm_size = monolis_mpi_get_global_comm_size();
  my_rank = monolis_mpi_get_local_my_rank(MPI_COMM_WORLD);

  if (comm_size > 1) {
    if (my_rank == 0) {
      for (i = 0; i < n_vertex; ++i) {
        global_index[i] = i;
      }
    } else {
      for (i = 0; i < n_vertex; ++i) {
        global_index[i] = 18 + (my_rank - 1) * 18 + i;
      }
    }

    monolis_com_initialize_by_global_id(&com, MPI_COMM_WORLD, 
                                        (my_rank == 0 || my_rank == comm_size - 1)? 18: 16, 
                                        n_vertex, global_index);
  } else {
    monolis_com_initialize_by_self(&com);
  }
  for (i = 0; i < n_vertex; ++i) {
    dof_list[i] = 1;
  }

  for (i = 0; i < n_vertex; ++i) {
    for (j = 0; j < n_vertex; ++j) {
      local_dense_matrix[i][j] = 0.0;
    }
  }
  for (i = 0; i < 20; ++i) {
    local_dense_matrix[i][i] = rand()%1001 + 5000.0;

    val = rand()%1001;
    if (i - 1 >= 0) {
      local_dense_matrix[i][i - 1] = val;
      local_dense_matrix[i - 1][i] = val;
    }
    val = rand()%1001;
    if (i + 1 < 20) {
      local_dense_matrix[i][i + 1] = val;
      local_dense_matrix[i + 1][i] = val;
    }
  }

  int index[21]; 
  int item[58];
  double fv[58];
  k = 0 ;
  for(i = 0; i < 20; ++i){
    index[i] = k;

    for(j = 0; j < 20; ++j){
      if(local_dense_matrix[i][j] != 0.0) {
	fv[k] = local_dense_matrix[i][j];
        item[k++] = j;
      }
    }
  }
  index[20] = k;

  monolis_set_matrix_BCSR_R(
      &mat, n_node * n_dof, n_node * n_dof, 1, 58,
      fv, index, item);

  for(i = 0; i < 20; ++i){
    a[i] = 1.0;
  }

  monolis_matvec_product_R(&mat, &com, a, b);

  for(i = 0; i < 20; ++i){
    a[i] = 0.0;
  }

  monolis_set_tolerance(&mat, 1.0e-10);

  monolis_solve_R(&mat, &com, b, a);

  monolis_mpi_update_R(&com, 10, 2, a);
  for(i = 0; i < 20; ++i){
    monolis_test_check_eq_R1("monolis_solve_c_test R", a[i], 1.0);
  }

  monolis_finalize(&mat);
}

void monolis_bcsr_solve_c_test(){
  monolis_BCSR_solve_c_test_R();
}
