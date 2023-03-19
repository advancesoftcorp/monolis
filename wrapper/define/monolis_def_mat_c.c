#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "monolis_utils.h"
#include "monolis_def_mat_c.h"

void monolis_mat_initialize_val_R(
  MONOLIS_MAT_VAL_R* mat)
{
  monolis_dealloc_R_1d(&mat->A);
  monolis_dealloc_R_1d(&mat->U);
  monolis_dealloc_R_1d(&mat->D);
  monolis_dealloc_R_1d(&mat->L);
  monolis_dealloc_R_1d(&mat->X);
  monolis_dealloc_R_1d(&mat->B);
}

void monolis_mat_initialize_val_C(
  MONOLIS_MAT_VAL_C* mat)
{
  monolis_dealloc_C_1d(&mat->A);
  monolis_dealloc_C_1d(&mat->U);
  monolis_dealloc_C_1d(&mat->D);
  monolis_dealloc_C_1d(&mat->L);
  monolis_dealloc_C_1d(&mat->X);
  monolis_dealloc_C_1d(&mat->B);
}

void monolis_mat_initialize_SCSR(
  MONOLIS_MAT_SEPARATED_CSR* mat)
{
  monolis_dealloc_I_1d(&mat->indexU);
  monolis_dealloc_I_1d(&mat->indexL);
  monolis_dealloc_I_1d(&mat->itemU);
  monolis_dealloc_I_1d(&mat->itemL);
}

void monolis_mat_initialize_CSR(
  MONOLIS_MAT_CSR* mat)
{
  monolis_dealloc_I_1d(&mat->index);
  monolis_dealloc_I_1d(&mat->item);
}

void monolis_mat_initialize_CSC(
  MONOLIS_MAT_CSC* mat)
{
  monolis_dealloc_I_1d(&mat->index);
  monolis_dealloc_I_1d(&mat->item);
  monolis_dealloc_I_1d(&mat->perm);
}

void monolis_mat_initialize(
  MONOLIS_MAT* mat)
{
  mat->N = 0;
  mat->NP = 0;
  mat->NDOF = 0;
  monolis_mat_initialize_val_R(&mat->R);
  monolis_mat_initialize_val_C(&mat->C);
  monolis_mat_initialize_SCSR(&mat->SCSR);
  monolis_mat_initialize_CSR(&mat->CSR);
  monolis_mat_initialize_CSC(&mat->CSC);
}

void monolis_mat_finalize_val_R(
  MONOLIS_MAT_VAL_R* mat)
{
  monolis_dealloc_R_1d(&mat->A);
  monolis_dealloc_R_1d(&mat->U);
  monolis_dealloc_R_1d(&mat->D);
  monolis_dealloc_R_1d(&mat->L);
  monolis_dealloc_R_1d(&mat->X);
  monolis_dealloc_R_1d(&mat->B);
}

void monolis_mat_finalize_val_C(
  MONOLIS_MAT_VAL_C* mat)
{
  monolis_dealloc_C_1d(&mat->A);
  monolis_dealloc_C_1d(&mat->U);
  monolis_dealloc_C_1d(&mat->D);
  monolis_dealloc_C_1d(&mat->L);
  monolis_dealloc_C_1d(&mat->X);
  monolis_dealloc_C_1d(&mat->B);
}

void monolis_mat_finalize_SCSR(
  MONOLIS_MAT_SEPARATED_CSR* mat)
{
  monolis_dealloc_I_1d(&mat->indexU);
  monolis_dealloc_I_1d(&mat->indexL);
  monolis_dealloc_I_1d(&mat->itemU);
  monolis_dealloc_I_1d(&mat->itemL);
}

void monolis_mat_finalize_CSR(
  MONOLIS_MAT_CSR* mat)
{
  monolis_dealloc_I_1d(&mat->index);
  monolis_dealloc_I_1d(&mat->item);
}

void monolis_mat_finalize_CSC(
  MONOLIS_MAT_CSC* mat)
{
  monolis_dealloc_I_1d(&mat->index);
  monolis_dealloc_I_1d(&mat->item);
  monolis_dealloc_I_1d(&mat->perm);
}

void monolis_mat_finalize(
  MONOLIS_MAT* mat)
{
  mat->N = 0;
  mat->NP = 0;
  mat->NDOF = 0;
  monolis_mat_finalize_val_R(&mat->R);
  monolis_mat_finalize_val_C(&mat->C);
  monolis_mat_finalize_SCSR(&mat->SCSR);
  monolis_mat_finalize_CSR(&mat->CSR);
  monolis_mat_finalize_CSC(&mat->CSC);
}