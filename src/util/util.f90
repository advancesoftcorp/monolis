module mod_monolis_util
  use mod_monolis_prm
  use mod_monolis_com
  use mod_monolis_mat

  implicit none

  private
  public :: monolis_initialize
  public :: monolis_finalize
  public :: monolis_timer_initialize
  public :: monolis_timer_finalize
  public :: monolis_check_diagonal

  !> tsol = tspmv + tprec + tcomm + others
  real(kind=kdouble) :: tsol  = 0.0d0
  real(kind=kdouble) :: tspmv = 0.0d0
  real(kind=kdouble) :: tprec = 0.0d0
  real(kind=kdouble) :: tcomm = 0.0d0

contains

  subroutine monolis_initialize(monoPRM, monoCOM, monoMAT)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT

    call monolis_prm_initialize(monoPRM)
    call monolis_com_initialize(monoCOM)
    call monolis_mat_initialize(monoMAT)
  end subroutine monolis_initialize

  subroutine monolis_finalize(monoPRM, monoCOM, monoMAT)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT

    call monolis_prm_finalize(monoPRM)
    call monolis_com_finalize(monoCOM)
    call monolis_mat_finalize(monoMAT)
  end subroutine monolis_finalize

  subroutine monolis_timer_initialize()
    implicit none

    tsol  = 0.0d0
    tspmv = 0.0d0
    tprec = 0.0d0
    tcomm = 0.0d0
  end subroutine monolis_timer_initialize

  subroutine monolis_timer_finalize(monoCOM)
    implicit none
    type(monolis_com) :: monoCOM

    !if(monoCOM%myrank == 0) write(*,"(a,i8,1p4e12.5)")" ** monolis solved:", 0, tsol, tspmv, tprec, tcomm
  end subroutine monolis_timer_finalize

  subroutine monolis_check_diagonal(monoMAT)
    implicit none
    type(monolis_mat) :: monoMAT
    integer(kind=kint) :: i, j, k, jS, jE, in, kn, NP, NDOF, NDOF2

    NP =  monoMAT%NP
    NDOF  = monoMAT%NDOF
    NDOF2 = NDOF*NDOF

    do i = 1, NP
      jS = monoMAT%index(i-1) + 1
      jE = monoMAT%index(i)
      do j = jS, jE
        in = monoMAT%item(j)
        if(i == in)then
          do k = 1, NDOF
            kn = NDOF2*(j-1) + (NDOF+1)*(k-1) + 1
            if(monoMAT%A(kn) == 0.0d0) stop " ** monolis error: zero diagonal"
          enddo
        endif
      enddo
    enddo
  end subroutine monolis_check_diagonal
end module mod_monolis_util