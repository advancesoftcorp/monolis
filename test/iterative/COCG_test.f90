!> COCG 法テストモジュール
module mod_monolis_solver_COCG_test
  use mod_monolis

  implicit none

contains

  subroutine monolis_solver_COCG_test()
    implicit none
    integer(kint) :: n_dof

    do n_dof = 1, 3
      call monolis_solver_COCG_test_main(n_dof, monolis_prec_NONE)
      call monolis_solver_COCG_test_main(n_dof, monolis_prec_DIAG)
      call monolis_solver_COCG_test_main(n_dof, monolis_prec_SOR)
    enddo
  end subroutine monolis_solver_COCG_test

  subroutine monolis_solver_COCG_test_main(n_dof, prec)
    implicit none
    type(monolis_structure) :: mat
    integer(kint) :: nnode, nelem, elem(2,9)
    integer(kint) :: i1, i2, j1, j2
    integer(kint) :: n_dof, prec
    real(kdouble) :: r1, r2
    complex(kdouble) :: val
    complex(kdouble) :: a(n_dof*10), b(n_dof*10)

    call monolis_std_log_string("monolis_solver_COCG_test_main")
    call monolis_std_log_I1("DOF", n_dof)
    call monolis_std_log_I1("PRECOND", prec)

    call monolis_initialize(mat, "./")

    nnode = 10

    nelem = 9

    elem(1,1) = 1; elem(2,1) = 2;
    elem(1,2) = 2; elem(2,2) = 3;
    elem(1,3) = 3; elem(2,3) = 4;
    elem(1,4) = 4; elem(2,4) = 5;
    elem(1,5) = 5; elem(2,5) = 6;
    elem(1,6) = 6; elem(2,6) = 7;
    elem(1,7) = 7; elem(2,7) = 8;
    elem(1,8) = 8; elem(2,8) = 9;
    elem(1,9) = 9; elem(2,9) =10;

    call monolis_get_nonzero_pattern_by_simple_mesh_C(mat, nnode, 2, n_dof, nelem, elem)

    do i1 = 1, 10
      do i2 = 1, n_dof
        call random_number(r1)
        call random_number(r2)
        val = cmplx(r1, r2) + (2.0d0, 2.0d0)
        call monolis_add_scalar_to_sparse_matrix_C(mat, i1, i1, i2, i2, val)
      enddo
    enddo

    do i1 = 1, 9
      do i2 = 1, n_dof
      do j2 = 1, n_dof
        call random_number(r1)
        call random_number(r2)
        val = cmplx(r1, r2)
        call monolis_add_scalar_to_sparse_matrix_C(mat, elem(1,i1), elem(2,i1), i2, j2, val)
        call monolis_add_scalar_to_sparse_matrix_C(mat, elem(2,i1), elem(1,i1), j2, i2, val)
      enddo
      enddo
    enddo

    a = (1.0d0, 1.0d0)

    call monolis_matvec_product_C(mat, a, b)

    call monolis_set_method(mat, monolis_iter_COCG)
    call monolis_set_precond(mat, prec)
    call monolis_set_tolerance(mat, 1.0d-12)
    call monolis_show_timelog_statistics(mat, .true.)

    call monolis_solve_C(mat, b, a)

    b = (1.0d0, 1.0d0)

    call monolis_test_check_eq_C("monolis_solver_COCG_test_main", a, b)

    call monolis_finalize(mat)
  end subroutine monolis_solver_COCG_test_main
end module mod_monolis_solver_COCG_test
