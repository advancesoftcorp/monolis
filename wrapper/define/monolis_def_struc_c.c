#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "monolis_utils.h"
#include "monolis_def_struc_c.h"

void monolis_com_input_comm_table(
  MONOLIS* mat,
  const char* top_dir_name,
  const char* part_dir_name,
  const char* file_name)
{
  if(mat->com.comm_size <= 1){
    mat->com.comm_size = 1;
    mat->com.send_n_neib = 0;
    mat->com.recv_n_neib = 0;
    return;
  }
}

void monolis_global_initialize()
{
  monolis_mpi_initialize();
}

void monolis_global_finalize()
{
  monolis_mpi_finalize();
}

void monolis_initialize(
  MONOLIS* mat)
{
printf("%s\n", "a");
  monolis_prm_initialize(&mat->prm);
printf("%s\n", "b");
  monolis_com_initialize(&mat->com);
printf("%s\n", "c");
  monolis_mat_initialize(&mat->mat);
printf("%s\n", "d");
  monolis_mat_initialize(&mat->prec);
printf("%s\n", "e");
  monolis_com_input_comm_table(mat,
    mat->prm.com_top_dir_name,
    mat->prm.com_part_dir_name,
    mat->prm.com_file_name);
printf("%s\n", "f");
}

void monolis_initialize_entire(
  MONOLIS* mat)
{
  monolis_prm_initialize(&mat->prm);
  monolis_com_initialize(&mat->com);
  monolis_mat_initialize(&mat->mat);
  monolis_mat_initialize(&mat->prec);

  mat->com.comm = 0;
  mat->com.my_rank = 0;
  mat->com.comm_size = 1;
}

void monolis_finalize(
  MONOLIS* mat)
{
  monolis_prm_initialize(&mat->prm);
  monolis_com_initialize(&mat->com);
  monolis_mat_initialize(&mat->mat);
  monolis_mat_initialize(&mat->prec);
}
