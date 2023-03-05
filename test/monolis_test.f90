program monolis_test
  use mod_monolis
  use mod_monolis_def_mat_test
  use mod_monolis_def_solver_test
  use mod_monolis_def_solver_util_test
  use mod_monolis_def_struc_test
  use mod_monolis_spmat_nonzero_pattern_util_test
  use mod_monolis_spmat_nonzero_pattern_test
  use mod_monolis_spmat_handler_util_test
  use mod_monolis_spmat_handler_test
  use mod_monolis_vec_util_test
  use mod_monolis_linalg_test
  use mod_monolis_converge_test
  use mod_monolis_matvec_test
  use mod_monolis_solver_CG_test
  use mod_monolis_solver_BiCGSTAB_test
  use mod_monolis_solver_COCG_test
  implicit none

  call monolis_global_initialize()

  call monolis_def_mat_test()
  call monolis_def_solver_test()
  call monolis_def_solver_util_test()
  call monolis_def_struc_test()

  call monolis_spmat_nonzero_pattern_util_test()
  call monolis_spmat_nonzero_pattern_test()
  call monolis_spmat_handler_util_test()
  call monolis_spmat_handler_test()

  call monolis_vec_util_test()
  call monolis_linalg_test()
  call monolis_converge_test()
  call monolis_matvec_test()

  call monolis_solver_CG_test()
  call monolis_solver_BiCGSTAB_test()
  call monolis_solver_COCG_test()

  call monolis_global_finalize()

end program monolis_test
