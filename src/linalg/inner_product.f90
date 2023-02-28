!> ベクトル内積関数群
module mod_monolis_linalg
  use mod_monolis_utils
  use mod_monolis_def_struc
  implicit none

contains

  !> @ingroup linalg
  !> ベクトル内積（整数型）
  subroutine monolis_inner_product_I(monolis, n, ndof, X, Y, sum)
    implicit none
    !> monolis 構造体
    type(monolis_structure) :: monolis
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    integer(kint) :: X(:)
    !> ベクトル 2
    integer(kint) :: Y(:)
    !> 内積結果
    integer(kint) :: sum
    real(kdouble) :: tdotp, tcomm

    call monolis_inner_product_main_I(monolis%COM, n, ndof, X, Y, sum, tdotp, tcomm)
  end subroutine monolis_inner_product_I

  !> @ingroup dev_linalg
  !> ベクトル内積（整数型、メイン関数）
  subroutine monolis_inner_product_main_I(monoCOM, n, ndof, X, Y, sum, tdotp, tcomm)
    implicit none
    !> monoCOM 構造体
    type(monolis_com) :: monoCOM
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    integer(kint) :: X(:)
    !> ベクトル 2
    integer(kint) :: Y(:)
    !> 内積結果
    integer(kint) :: sum
    !> 内積時間
    real(kdouble), optional :: tdotp
    !> 通信時間
    real(kdouble), optional :: tcomm
    integer(kint) :: i
    real(kdouble) :: t1, t2, t3

#ifdef DEBUG
    call monolis_std_debug_log_header("monolis_inner_product_main_I")
#endif

    t1 = monolis_get_time()
    sum = 0
!$omp parallel default(none) &
!$omp & shared(X, Y, sum) &
!$omp & firstprivate(n, ndof) &
!$omp & private(i)
!$omp do reduction(+:sum)
    do i = 1, n * ndof
      sum = sum + X(i)*Y(i)
    enddo
!$omp end do
!$omp end parallel

    t2 = monolis_get_time()
    call monolis_allreduce_I1(sum, monolis_mpi_sum, monoCOM%comm)
    t3 = monolis_get_time()

    if(present(tdotp)) tdotp = tdotp + t3 - t1
    if(present(tcomm)) tcomm = tcomm + t3 - t2
  end subroutine monolis_inner_product_main_I

  !> @ingroup linalg
  !> ベクトル内積（実数型）
  subroutine monolis_inner_product_R(monolis, n, ndof, X, Y, sum)
    implicit none
    !> monolis 構造体
    type(monolis_structure) :: monolis
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    real(kdouble) :: X(:)
    !> ベクトル 2
    real(kdouble) :: Y(:)
    !> 内積結果
    real(kdouble) :: sum
    real(kdouble) :: tdotp, tcomm

    call monolis_inner_product_main_R(monolis%COM, n, ndof, X, Y, sum, tdotp, tcomm)
  end subroutine monolis_inner_product_R

  !> @ingroup dev_linalg
  !> ベクトル内積（実数型、メイン関数）
  subroutine monolis_inner_product_main_R(monoCOM, n, ndof, X, Y, sum, tdotp, tcomm)
    implicit none
    !> monoCOM 構造体
    type(monolis_com) :: monoCOM
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    real(kdouble) :: X(:)
    !> ベクトル 2
    real(kdouble) :: Y(:)
    !> 内積結果
    real(kdouble) :: sum
    !> 内積時間
    real(kdouble), optional :: tdotp
    !> 通信時間
    real(kdouble), optional :: tcomm
    integer(kint) :: i
    real(kdouble) :: t1, t2, t3

#ifdef DEBUG
    call monolis_std_debug_log_header("monolis_inner_product_main_R")
#endif

    t1 = monolis_get_time()
    sum = 0.0d0
!$omp parallel default(none) &
!$omp & shared(X, Y, sum) &
!$omp & firstprivate(n, ndof) &
!$omp & private(i)
!$omp do reduction(+:sum)
    do i = 1, n * ndof
      sum = sum + X(i)*Y(i)
    enddo
!$omp end do
!$omp end parallel

    t2 = monolis_get_time()
    call monolis_allreduce_R1(sum, monolis_mpi_sum, monoCOM%comm)
    t3 = monolis_get_time()

    if(present(tdotp)) tdotp = tdotp + t3 - t1
    if(present(tcomm)) tcomm = tcomm + t3 - t2
  end subroutine monolis_inner_product_main_R

  !> @ingroup linalg
  !> ベクトル内積（複素数型）
  subroutine monolis_inner_product_C(monolis, n, ndof, X, Y, sum)
    implicit none
    !> monolis 構造体
    type(monolis_structure) :: monolis
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    complex(kdouble) :: X(:)
    !> ベクトル 2
    complex(kdouble) :: Y(:)
    !> 内積結果
    complex(kdouble) :: sum
    real(kdouble) :: tdotp, tcomm

    call monolis_inner_product_main_C(monolis%COM, n, ndof, X, Y, sum, tdotp, tcomm)
  end subroutine monolis_inner_product_C

  !> @ingroup dev_linalg
  !> ベクトル内積（複素数型、メイン関数）
  subroutine monolis_inner_product_main_C(monoCOM, n, ndof, X, Y, sum, tdotp, tcomm)
    implicit none
    !> monoCOM 構造体
    type(monolis_com) :: monoCOM
    !> 自由度数
    integer(kint) :: n
    !> ブロックサイズ
    integer(kint) :: ndof
    !> ベクトル 1
    complex(kdouble) :: X(:)
    !> ベクトル 2
    complex(kdouble) :: Y(:)
    !> 内積結果
    complex(kdouble) :: sum
    !> 内積時間
    real(kdouble), optional :: tdotp
    !> 通信時間
    real(kdouble), optional :: tcomm
    integer(kint) :: i
    real(kdouble) :: t1, t2, t3

#ifdef DEBUG
    call monolis_std_debug_log_header("monolis_inner_product_main_C")
#endif

    t1 = monolis_get_time()
    sum = 0.0d0
!$omp parallel default(none) &
!$omp & shared(X, Y, sum) &
!$omp & firstprivate(n, ndof) &
!$omp & private(i)
!$omp do reduction(+:sum)
    do i = 1, n * ndof
      sum = sum + X(i)*Y(i)
    enddo
!$omp end do
!$omp end parallel

    t2 = monolis_get_time()
    call monolis_allreduce_C1(sum, monolis_mpi_sum, monoCOM%comm)
    t3 = monolis_get_time()

    if(present(tdotp)) tdotp = tdotp + t3 - t1
    if(present(tcomm)) tcomm = tcomm + t3 - t2
  end subroutine monolis_inner_product_main_C

  !> @ingroup dev_linalg
  !> ベクトル内積（実数型、メイン関数、通信なし）
  subroutine monolis_inner_product_main_R_no_comm(monoCOM, n, ndof, X, Y, sum)
    implicit none
    type(monolis_com) :: monoCOM
    integer(kint) :: i, n, ndof
    real(kdouble) :: X(:), Y(:)
    real(kdouble) :: sum

    sum = 0.0d0
!$omp parallel default(none) &
!$omp & shared(X, Y, sum) &
!$omp & firstprivate(n, ndof) &
!$omp & private(i)
!$omp do reduction(+:sum)
    do i = 1, n * ndof
      sum = sum + X(i)*Y(i)
    enddo
!$omp end do
!$omp end parallel
  end subroutine monolis_inner_product_main_R_no_comm
end module mod_monolis_linalg