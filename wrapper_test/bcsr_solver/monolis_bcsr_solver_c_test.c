#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <complex.h>
#include "monolis.h"

void monolis_BCSR_solve_c_test_R(){
  MONOLIS mat;
  MONOLIS_COM com;
  int n_node;
  int n_base;
  int n_dof;
  int n_elem;
  int i, j, k;
  int** elem;
  double val;
  double a[20];
  double b[20];

  monolis_std_log_string("monolis_solve_c_test_R");

  n_node = 10;
  n_base = 2;
  n_dof  = 2;
  n_elem = 9;

  elem = monolis_alloc_I_2d(elem, n_node, n_base);

  elem[0][0] = 0; elem[0][1] = 1;
  elem[1][0] = 1; elem[1][1] = 2;
  elem[2][0] = 2; elem[2][1] = 3;
  elem[3][0] = 3; elem[3][1] = 4;
  elem[4][0] = 4; elem[4][1] = 5;
  elem[5][0] = 5; elem[5][1] = 6;
  elem[6][0] = 6; elem[6][1] = 7;
  elem[7][0] = 7; elem[7][1] = 8;
  elem[8][0] = 8; elem[8][1] = 9;

  monolis_initialize(&mat);
  monolis_com_initialize_by_self(&com);

  double matrix[20][20];
  for (i = 0; i < 20; ++i) {
    for (j = 0; j < 20; ++j) {
      matrix[i][j] = 0.0;
    }
  }
  for (i = 0; i < 20; ++i) {
    matrix[i][i] = rand()%1001 + 5000.0;

    val = rand()%1001;
    if (i - 1 >= 0) {
      matrix[i][i - 1] = val;
      matrix[i - 1][i] = val;
    }
    val = rand()%1001;
    if (i + 1 < 20) {
      matrix[i][i + 1] = val;
      matrix[i + 1][i] = val;
    }
  }

  int index[21]; 
  int item[58];
  double fv[58];
  k = 0 ;
  for(i = 0; i < 20; ++i){
    index[i] = k;

    for(j = 0; j < 20; ++j){
      if(matrix[i][j] != 0.0) {
	fv[k] = matrix[i][j];
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

  for(i = 0; i < 20; ++i){
    monolis_test_check_eq_R1("monolis_solve_c_test R", a[i], 1.0);
  }

  monolis_finalize(&mat);
}

void monolis_bcsr_solve_c_test(){
  monolis_BCSR_solve_c_test_R();
}
