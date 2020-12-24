module mod_monolis_eigen_lanczos_util
  use mod_monolis_prm
  use mod_monolis_com
  use mod_monolis_mat
  use mod_monolis_linalg

  implicit none

contains

  subroutine lanczos_initialze(n, q)
    implicit none
    integer(kint) :: i, n
    real(kdouble) :: q(:), norm

    norm = 0.0d0
    do i = 1, n
      q(i) = dble(i)
      norm = norm + q(i)*q(i)
    enddo

    norm = 1.0d0/dsqrt(norm)
    do i = 1, n
      q(i) = q(i)*norm
    enddo
  end subroutine lanczos_initialze

  subroutine monolis_gram_schmidt(monoPRM, monoCOM, monoMAT, iter, q, p)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    integer(kint) :: i, j, iter, N, NDOF
    real(kdouble) :: q(:,0:), p(:), norm

    N    = monoMAT%N
    NDOF = monoMAT%NDOF

    do i = 1, iter-1
      call monolis_inner_product_R(monoCOM, N, NDOF, p, q(:,i), norm, monoPRM%tdotp, monoPRM%tcomm_dotp)

      do j = 1, N*NDOF
        p(j) = p(j) - norm*q(j,i)
      enddo
    enddo
  end subroutine monolis_gram_schmidt

  subroutine monolis_get_eigen_pair_from_tridiag(iter, alpha_t, beta_t, q, e_value, e_mode)
    implicit none
    integer(kint) :: iter, i, n, m, iu, il, ldz, info, liwork, lwork
    real(kdouble) :: alpha_t(:), beta_t(:), q(:,0:), e_value(:), e_mode(:,:)
    real(kdouble) :: vl, vu, abstol
    integer(kint), allocatable :: isuppz(:), idum(:)
    real(kdouble), allocatable :: alpha(:), beta(:), rdum(:), e_mode_t(:,:)

    !> DSTEVR
    allocate(alpha(iter), source = 0.0d0)
    allocate(beta (max(1,iter-1)), source = 0.0d0)
    allocate(isuppz(2*iter), source = 0)
    allocate(idum(10*iter), source = 0)
    allocate(rdum(20*iter), source = 0.0d0)
    allocate(e_mode_t(iter,iter), source = 0.0d0)

    alpha = alpha_t(1:iter)
    beta = beta_t(2:max(1,iter-1)+1)

    vl = 0.0d0
    vu = 0.0d0
    il = 0
    iu = 0
    abstol = 1.0d-8
    n = iter
    m = iter
    ldz = iter
    lwork = 20*iter
    liwork = 10*iter

    call dstevr("V", "A", n, alpha, beta, vl, vu, il, iu, abstol, m, e_value, &
      e_mode_t, ldz, isuppz, rdum, lwork, idum, liwork, info)
    if(info /= 0) stop "monolis_get_eigen_pair_from_tridiag"

    write(*,*)"e_value"
    do i = 1, iter
      write(*,"(1p2e12.5)")e_value(i), 1.0d0/e_value(i)
    enddo

    !e_mode(:,1:iter) = matmul(q(:,1:iter), transpose(e_mode_t))
    !write(*,*)"e_mode"
    !do i = 1, iter
      !write(*,"(1p10e12.5)")e_mode(:,i)
      !write(*,"(1p10e12.5)")e_mode_t(:,i)
      !write(*,"(1p10e12.5)")q(:,i)
    !enddo

    deallocate(alpha)
    deallocate(beta)
    deallocate(isuppz)
    deallocate(idum)
    deallocate(e_mode_t)
    deallocate(rdum)
  end subroutine monolis_get_eigen_pair_from_tridiag
end module mod_monolis_eigen_lanczos_util
