module mod_monolis_solver_GropCG
  use mod_monolis_prm
  use mod_monolis_com
  use mod_monolis_mat
  use mod_monolis_precond
  use mod_monolis_matvec
  use mod_monolis_linalg
  use mod_monolis_linalg_util

  implicit none

contains

  subroutine monolis_solver_GropCG(monoPRM, monoCOM, monoMAT)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    integer(kind=kint) :: N, NP, NDOF, NNDOF
    integer(kind=kint) :: i, j, k, l, iter, iter_RR
    integer(kind=kint) :: reqs1(1)
    integer(kind=kint) :: reqs2(1)
    integer(kind=kint) :: statuses(monolis_status_size,1)
    real(kind=kdouble) :: tol, resid, R2, B2
    real(kind=kdouble) :: t1, t2, tset, tsol, tcomm
    real(kind=kdouble) :: alpha, beta, delta, gamma, gamma1, omega
    real(kind=kdouble) :: buf1(1), buf2(2)
    real(kind=kdouble), allocatable :: W(:,:)
    real(kind=kdouble), pointer :: B(:), X(:)
    integer(kind=kint), parameter :: R = 1
    integer(kind=kint), parameter :: U = 2
    integer(kind=kint), parameter :: V = 3
    integer(kind=kint), parameter :: Q = 4
    integer(kind=kint), parameter :: P = 5
    integer(kind=kint), parameter :: S = 6

    t1 = monolis_wtime()

    N     = monoMAT%N
    NP    = monoMAT%NP
    NDOF  = monoMAT%NDOF
    NNDOF = N*NDOF
    X => monoMAT%X; X = 1.0d0
    B => monoMAT%B

    allocate(W(NDOF*NP, 6))
    W = 0.0d0

    iter_RR = 50
    tol = monoPRM%tol

    call monolis_precond_setup(monoPRM, monoCOM, monoMAT)
    call monolis_residual(monoCOM, monoMAT, X, B, W(:,R), tcomm)
    call monolis_inner_product_R(monoCOM, monoMAT, NDOF, B, B, B2, tcomm)
    call monolis_precond_apply(monoPRM, monoCOM, monoMAT, W(:,R), W(:,U))

    do i = 1,NDOF*NP
      W(i,P) = W(i,U)
    enddo

    call monolis_matvec(monoCOM, monoMAT, W(:,P), W(:,S), tcomm)
    call monolis_inner_product_R(monoCOM, monoMAT, NDOF, W(:,R), W(:,U), gamma, tcomm)

    do iter = 1, monoPRM%maxiter
      call monolis_inner_product_R(monoCOM, monoMAT, NDOF, W(:,P), W(:,S), delta, tcomm)
      call monolis_precond_apply(monoPRM, monoCOM, monoMAT, W(:,S), W(:,Q))

      alpha = gamma/delta

      do i = 1, NNDOF
        X(i)   = X(i)   + alpha * W(i,P)
        W(i,R) = W(i,R) - alpha * W(i,S)
        W(i,U) = W(i,U) - alpha * W(i,Q)
      enddo

      call monolis_inner_product_R(monoCOM, monoMAT, NDOF, W(:,R), W(:,U), gamma1, tcomm)
      call monolis_inner_product_R(monoCOM, monoMAT, NDOF, W(:,R), W(:,R), R2, tcomm)

      call monolis_matvec(monoCOM, monoMAT, W(:,U), W(:,V), tcomm)

      beta  = gamma1/gamma
      gamma = gamma1

      do i = 1, NNDOF
        W(i,P) = W(i,U) + beta * W(i,P)
        W(i,S) = W(i,V) + beta * W(i,S)
      enddo

      resid = dsqrt(R2/B2)
      if(monoCOM%myrank == 0) write (*,"(i7, 1pe16.6)") iter, resid
      if(resid <= tol) exit
    enddo

    call monolis_update_R(monoCOM, NDOF, X, tcomm)
    call monolis_precond_clear(monoPRM, monoCOM, monoMAT)

    deallocate(W)

    t2 = monolis_wtime()
    tsol = t2 - t1
  end subroutine monolis_solver_GropCG

end module mod_monolis_solver_GropCG
