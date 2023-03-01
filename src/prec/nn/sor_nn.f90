!> SOR 前処理（nxn ブロック）
module mod_monolis_precond_sor_nn
  use mod_monolis_utils
  use mod_monolis_def_mat
  use mod_monolis_def_struc

  implicit none

contains

  !> 前処理生成：SOR 前処理（nxn ブロック）
  subroutine monolis_precond_sor_nn_setup(monoMAT, monoPREC)
    implicit none
    !> 行列構造体
    type(monolis_mat), target :: monoMAT
    !> 前処理構造体
    type(monolis_mat), target :: monoPREC
    integer(kint) :: i, ii, j, jS, jE, in, k, l, N, NDOF, NDOF2
    integer(kint), pointer :: index(:), item(:)
    real(kdouble) :: sigma
    real(kdouble), allocatable :: T(:), LU(:,:)
    real(kdouble), pointer :: A(:), ALU(:)

    N =  monoMAT%N
    NDOF  = monoMAT%NDOF
    NDOF2 = NDOF*NDOF
    A => monoMAT%R%A
    index => monoMAT%CSR%index
    item => monoMAT%CSR%item
    sigma = 1.0d0

    call monolis_alloc_R_1d(T, NDOF)
    call monolis_alloc_R_2d(LU, NDOF, NDOF)
    call monolis_alloc_R_1d(monoPREC%R%D, NDOF2*N)
    ALU => monoPREC%R%D

!$omp parallel default(none) &
!$omp & shared(A, ALU, index, item) &
!$omp & firstprivate(N, NDOF, NDOF2) &
!$omp & private(T, LU, i, j, k, jS, jE, in)
!$omp do
    do i = 1, N
      jS = index(i-1) + 1
      jE = index(i)
      do ii = jS, jE
        in = item(ii)
        if(i == in)then
          do j = 1, NDOF
            do k = 1, NDOF
              LU(j,k) = A(NDOF2*(ii-1) + NDOF*(j-1) + k)
            enddo
          enddo
          do k = 1, NDOF
            if(LU(k,k) == 0.0d0) stop "** monolis error: zero diag in monolis_precond_sor_nn_setup"
            LU(k,k) = 1.0d0/LU(k,k)
            do l = k+1, NDOF
              LU(l,k) = LU(l,k)*LU(k,k)
              do j = k+1, NDOF
                T(j) = LU(l,j) - LU(l,k)*LU(k,j)
              enddo
              do j = k+1, NDOF
                LU(l,j) = T(j)
              enddo
            enddo
          enddo
          do j = 1, NDOF
            do k = 1, NDOF
              ALU(NDOF2*(i-1) + NDOF*(j-1) + k) = LU(j,k)
            enddo
          enddo
        endif
      enddo
    enddo
!$omp end do
!$omp end parallel

    deallocate(T)
    deallocate(LU)
  end subroutine monolis_precond_sor_nn_setup

  !> 前処理適用：SOR 前処理（3x3 ブロック）
  subroutine monolis_precond_sor_nn_apply(monoMAT, monoPREC, X, Y)
    implicit none
    !> 行列構造体
    type(monolis_mat), target :: monoMAT
    !> 前処理構造体
    type(monolis_mat), target :: monoPREC
    integer(kint) :: i, j, jE, jS, jn, k, l, N, NP, NDOF, NDOF2
    integer(kint), pointer :: index(:)
    integer(kint), pointer :: item(:)
    real(kdouble) :: X(:), Y(:)
    real(kdouble), pointer :: A(:), ALU(:)
    real(kdouble), allocatable :: XT(:), YT(:), ST(:)

    N =  monoPREC%N
    NP = monoMAT%NP
    NDOF = monoMAT%NDOF
    NDOF2 = NDOF*NDOF
    ALU => monoPREC%R%D
    index => monoMAT%CSR%index
    item => monoMAT%CSR%item

!$omp parallel default(none) &
!$omp & shared(NP, NDOF, X, Y) &
!$omp & private(i)
!$omp do
    do i = 1, NP*NDOF
      Y(i) = X(i)
    enddo
!$omp end do
!$omp end parallel

    call monolis_alloc_R_1d(XT, NDOF)
    call monolis_alloc_R_1d(YT, NDOF)
    call monolis_alloc_R_1d(ST, NDOF)

    do i = 1, N
      do j = 1, NDOF
        ST(j) = Y(NDOF*(i-1)+j)
      enddo
      jS = index(i-1) + 1
      jE = index(i  )
      do j = jS, jE
        jn = item(j)
        if(jn < i)then
          do k = 1, NDOF
            XT(k) = Y(NDOF*(jn-1)+k)
          enddo
          do k = 1, NDOF
            do l = 1, NDOF
              ST(k) = ST(k) - A(NDOF2*(j-1)+NDOF*(k-1)+l) * XT(l)
            enddo
          enddo
        endif
      enddo

      do j = 1, NDOF
        XT(j) = ST(j)
      enddo
      do j = 2, NDOF
        do k = 1, j-1
          XT(j) = XT(j) - ALU(NDOF2*(i-1) + NDOF*(j-1) + k)*XT(k)
        enddo
      enddo
      do j = NDOF, 1, -1
        do k = NDOF, j+1, -1
          XT(j) = XT(j) - ALU(NDOF2*(i-1) + NDOF*(j-1) + k)*XT(k)
        enddo
        XT(j) = ALU(NDOF2*(i-1) + (NDOF+1)*(j-1) + 1)*XT(j)
      enddo
      do k = 1, NDOF
        Y(NDOF*(i-1)+k) = XT(k)
      enddo
    enddo

    do i = N, 1, -1
      do j = 1, NDOF
        ST(j) = 0.0d0
      enddo
      jS = index(i-1) + 1
      jE = index(i  )
      do j = jE, jS, -1
        jn = item(j)
        if(i < jn)then
          do k = 1, NDOF
            XT(k) = Y(NDOF*(jn-1)+k)
          enddo
          do k = 1, NDOF
            do l = 1, NDOF
              ST(k) = ST(k) + A(NDOF2*(j-1)+(k-1)*NDOF+l)*XT(l)
            enddo
          enddo
        endif
      enddo

      do j = 1, NDOF
        XT(j) = ST(j)
      enddo
      do j = 2, NDOF
        do k = 1, j-1
          XT(j) = XT(j) - ALU(NDOF2*(i-1) + NDOF*(j-1) + k)*XT(k)
        enddo
      enddo
      do j = NDOF, 1, -1
        do k = NDOF, j+1, -1
          XT(j) = XT(j) - ALU(NDOF2*(i-1) + NDOF*(j-1) + k)*XT(k)
        enddo
        XT(j) = ALU(NDOF2*(i-1) + (NDOF+1)*(j-1) + 1)*XT(j)
      enddo
      do k = 1, NDOF
        Y(NDOF*(i-1)+k) = Y(NDOF*(i-1)+k) - XT(k)
      enddo
    enddo

    deallocate(XT)
    deallocate(YT)
    deallocate(ST)
  end subroutine monolis_precond_sor_nn_apply

  !> 前処理初期化：SOR 前処理（nxn ブロック）
  subroutine monolis_precond_sor_nn_clear(monoPREC)
    implicit none
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_dealloc_R_1d(monoPREC%R%D)
  end subroutine monolis_precond_sor_nn_clear

end module mod_monolis_precond_sor_nn
