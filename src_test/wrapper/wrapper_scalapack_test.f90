!> ScaLAPACK ラッパーモジュール
module mod_monolis_scalapack_test
  use mod_monolis_utils
  use mod_monolis_scalapack

  implicit none

contains

  subroutine monolis_scalapack_test()
    implicit none
    !> 行列の大きさ（行数 N）
    integer(kint) :: N_loc = 4
    !> 行列の大きさ（列数 M）
    integer(kint) :: M = 3
    !> 入力行列（N_loc x M）
    real(kdouble) :: A(4,3)
    !> 左特異行列（N_loc x P）
    real(kdouble) :: S(4,3)
    !> 特異値（P）
    real(kdouble) :: V(3)
    !> 右特異行列（P x M）
    real(kdouble) :: D(3,3)
    integer(kint) :: comm

    real(kdouble) :: Vt(3,3)
    real(kdouble) :: VD(3,3)

    call monolis_std_global_log_string("monolis_scalapack_gesvd_R")

    comm = monolis_mpi_get_global_comm()

    A = 0.0d0
    S = 0.0d0
    V = 0.0d0
    D = 0.0d0

    A(1,1) = 1.0d0
    A(2,1) = 2.0d0
    A(3,1) = 3.0d0
    A(4,1) = 4.0d0

    A(1,2) = 11.0d0
    A(2,2) = 12.0d0
    A(3,2) = 13.0d0
    A(4,2) = 14.0d0

    A(1,3) = 21.0d0
    A(2,3) = 22.0d0
    A(3,3) = 23.0d0
    A(4,3) = 24.0d0

    call monolis_scalapack_gesvd_R(N_loc, M, A, S, V, D, comm)

    Vt = 0.0d0
    Vt(1,1) = V(1)
    Vt(2,2) = V(2)
    Vt(3,3) = V(3)

    !write(*,*)"S"
    !write(*,"(1pe12.4)")S
    !write(*,*)"V"
    !write(*,"(1pe12.4)")Vt
    !write(*,*)"D"
    !write(*,"(1pe12.4)")D

    write(*,*)"A"
    VD = matmul(Vt, D)
    write(*,"(1pe12.4)")matmul(S,VD)

  end subroutine monolis_scalapack_test

end module mod_monolis_scalapack_test
