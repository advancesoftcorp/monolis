/* monolis_def_struc.h */
#ifndef MONOLIS_DEF_SOLVER_H
#define MONOLIS_DEF_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif

void monolis_set_method(
  MONOLIS* mat,
  int      param);

void monolis_set_precond(
  MONOLIS* mat,
  int      param);

void monolis_set_maxiter(
  MONOLIS* mat,
  int      param);

void monolis_get_converge_iter(
  MONOLIS* mat,
  int*     param);

void monolis_get_error_tag(
  MONOLIS* mat,
  int*     param);

void monolis_set_init_x(
  MONOLIS* mat,
  bool     param);

void monolis_set_sym_matrix(
  MONOLIS* mat,
  bool     param);

void monolis_set_debug(
  MONOLIS* mat,
  bool     param);

void monolis_set_performance_measurement(
  MONOLIS* mat,
  bool     param);

void monolis_set_check_diag(
  MONOLIS* mat,
  bool     param);

void monolis_set_prec_stored(
  MONOLIS* mat,
  bool     param);

void monolis_show_iterlog(
  MONOLIS* mat,
  bool     param);

void monolis_show_timelog(
  MONOLIS* mat,
  bool     param);

void monolis_show_summary(
  MONOLIS* mat,
  bool     param);

void monolis_show_timelog_statistics(
  MONOLIS* mat,
  bool     param);

void monolis_set_tolerance(
  MONOLIS* mat,
  double   val);

void monolis_get_converge_residual(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_solver(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_preparing(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_spmv(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_inner_product(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_precondition(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_comm_inner_product(
  MONOLIS* mat,
  double*  val);

void monolis_get_time_comm_spmv(
  MONOLIS* mat,
  double*  val);

#ifdef __cplusplus
}
#endif

#endif
