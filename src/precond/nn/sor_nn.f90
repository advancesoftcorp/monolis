module mod_monolis_precond_sor_nn
  use mod_monolis_prm
  use mod_monolis_com
  use mod_monolis_mat

  implicit none

  private
  public :: monolis_precond_sor_nn_setup
  public :: monolis_precond_sor_nn_apply
  public :: monolis_precond_sor_nn_clear

  real(kind=kdouble), pointer :: ALU(:) => null()

contains

  subroutine monolis_precond_sor_nn_setup(monoMAT)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    integer(kind=kint) :: i, ii, j, jS, jE, in, k, l, N, NDOF, NDOF2
    integer(kind=kint), pointer :: index(:), item(:)
    real(kind=kdouble) :: sigma
    real(kind=kdouble), allocatable :: T(:), LU(:,:)
    real(kind=kdouble), pointer :: A(:)

    N =  monoMAT%N
    NDOF  = monoMAT%NDOF
    NDOF2 = NDOF*NDOF
    A => monoMAT%A
    index => monoMAT%index
    item => monoMAT%item
    sigma = 1.0d0

    allocate(T(NDOF))
    allocate(LU(NDOF,NDOF))
    allocate(ALU(NDOF2*N))
    T   = 0.0d0
    ALU = 0.0d0
    LU  = 0.0d0

    do i = 1, N
      jS = index(i-1) + 1
      jE = index(i)
      do ii = jS, jE
        in = item(ii)
        if(i == in)then
          do j = 1, NDOF
            do k = 1, NDOF
              LU(j,k) = A(NDOF2*(i-1) + NDOF*(j-1) + k)
            enddo
          enddo

          do k = 1, NDOF
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

    deallocate(T)
    deallocate(LU)
  end subroutine monolis_precond_sor_nn_setup

  subroutine monolis_precond_sor_nn_apply(monoMAT, X, Y)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    integer(kind=kint) :: i, j, jE, jS, jn, k, l, N, NP, NDOF, NDOF2
    integer(kind=kint), pointer :: index(:)
    integer(kind=kint), pointer :: item(:)
    real(kind=kdouble) :: X(:), Y(:)
    real(kind=kdouble), pointer :: A(:)
    real(kind=kdouble), allocatable :: XT(:), YT(:), ST(:)

    N     = monoMAT%N
    NP    = monoMAT%NP
    NDOF  = monoMAT%NDOF
    NDOF2 = NDOF*NDOF
    index => monoMAT%index
    item => monoMAT%item
    A => monoMAT%A

    do i = 1, NP*NDOF
      Y(i) = X(i)
    enddo

    allocate(XT(NDOF))
    allocate(YT(NDOF))
    allocate(ST(NDOF))
    XT = 0.0d0
    YT = 0.0d0
    ST = 0.0d0

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

  subroutine monolis_precond_sor_nn_clear()
    implicit none
    deallocate(ALU)
  end subroutine monolis_precond_sor_nn_clear

end module mod_monolis_precond_sor_nn
