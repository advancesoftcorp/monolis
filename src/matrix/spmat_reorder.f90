module mod_monolis_reorder
  use mod_monolis_utils
  use mod_monolis_def_mat
  use mod_monolis_def_struc

  implicit none

  type monolis_edge_info
    integer(kint) :: N = 0
    integer(kint), pointer :: node(:) => null()
  endtype monolis_edge_info

contains

  subroutine monolis_reorder_matrix_fw(monoPRM, monoCOM, monoCOM_reorder, monoMAT, monoMAT_reorder)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_com) :: monoCOM_reorder
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoMAT_reorder
    integer(kint), pointer ::  perm(:), iperm(:)
    real(kdouble) :: t1, t2

!    if(monoPRM%is_debug) call monolis_std_debug_log_header("monolis_reorder_matrix_fw")
!    t1 = monolis_get_time()
!
!    if(monoPRM%is_reordering)then
!#ifdef WITH_METIS
!      allocate(monoMAT%perm(monoMAT%NP))
!      allocate(monoMAT%iperm(monoMAT%NP))
!      perm => monoMAT%perm
!      iperm => monoMAT%iperm
!      call monolis_reorder_matrix_metis(monoMAT, monoMAT_reorder)
!      call monolis_restruct_matrix(monoMAT, monoMAT_reorder, perm, iperm)
!      call monolis_restruct_comm(monoCOM, monoCOM_reorder, iperm)
!      call monolis_reorder_vector_fw(monoMAT, monoMAT%NP, monoMAT%NDOF, monoMAT%B, monoMAT_reorder%B)
!      if(.not. monoPRM%is_init_x)then
!        call monolis_reorder_vector_fw(monoMAT, monoMAT%NP, monoMAT%NDOF, monoMAT%X, monoMAT_reorder%X)
!      endif
!#else
!      call monolis_copy_mat_by_pointer(monoMAT, monoMAT_reorder)
!      call monolis_com_copy(monoCOM, monoCOM_reorder)
!#endif
!    else
!      call monolis_copy_mat_by_pointer(monoMAT, monoMAT_reorder)
!      call monolis_com_copy(monoCOM, monoCOM_reorder)
!    endif
!
!    t2 = monolis_get_time()
!    monoPRM%tprep = monoPRM%tprep + t2 - t1
  end subroutine monolis_reorder_matrix_fw

  subroutine monolis_reorder_matrix_bk(monoPRM, monoCOM, monoMAT_reorder, monoMAT)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoMAT_reorder
    real(kdouble) :: t1, t2

!    if(monoPRM%is_debug) call monolis_std_debug_log_header("monolis_reorder_matrix_bk")
!    t1 = monolis_get_time()
!
!    if(monoPRM%is_reordering)then
!#ifdef WITH_METIS
!      call monolis_reorder_back_vector_bk(monoMAT, monoMAT%NP, monoMAT%NDOF, monoMAT_reorder%X, monoMAT%X)
!      deallocate(monoMAT%perm)
!      deallocate(monoMAT%iperm)
!#endif
!    endif
!
!    t2 = monolis_get_time()
!    monoPRM%tprep = monoPRM%tprep + t2 - t1
  end subroutine monolis_reorder_matrix_bk

  subroutine monolis_reorder_vector_fw(monoMAT, N, NDOF, A, B)
    implicit none
    type(monolis_mat) :: monoMAT
    integer(kint) :: N, NDOF
    real(kdouble) :: A(:)
    real(kdouble) :: B(:)
    integer(kint) :: i, in, jn, jo, j
!    do i = 1, N
!      in = monoMAT%iperm(i)
!      jn = (in-1)*NDOF
!      jo = (i -1)*NDOF
!      do j = 1, NDOF
!        B(jn + j) = A(jo + j)
!      enddo
!    enddo
  end subroutine monolis_reorder_vector_fw

  subroutine monolis_reorder_back_vector_bk(monoMAT, N, NDOF, B, A)
    implicit none
    type(monolis_mat) :: monoMAT
    integer(kint) :: N, NDOF
    real(kdouble) :: B(:)
    real(kdouble) :: A(:)
    integer(kint) :: i, in, jn, jo, j
!    do i = 1, N
!      in = monoMAT%perm(i)
!      jn = (i -1)*NDOF
!      jo = (in-1)*NDOF
!      do j = 1, NDOF
!        A(jo + j) = B(jn + j)
!      enddo
!    enddo
  end subroutine monolis_reorder_back_vector_bk

  subroutine monolis_reorder_matrix_metis(monoMAT, monoMAT_reorder)
    implicit none
    type(monolis_edge_info), allocatable :: edge(:)
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoMAT_reorder
    integer(kint) :: N, NP
    integer(kint) :: i, j, jS, jE
    integer(kint) :: nedge, in, jn
    integer(kint), allocatable :: nozero(:)
    integer(kint) :: nvtxs
    integer(kint), pointer :: index(:)   => null()
    integer(kint), pointer :: item(:)    => null()
    integer(kint), pointer :: xadj(:)    => null()
    integer(kint), pointer :: adjncy(:)  => null()
    integer(kint), pointer :: vwgt(:)    => null()
    integer(kint), pointer :: options(:) => null()
    integer(kint), pointer :: perm(:), iperm(:)

!    N = monoMAT%N
!    NP = monoMAT%NP
!    index => monoMAT%index
!    item  => monoMAT%item
!    perm => monoMAT%perm
!    iperm => monoMAT%iperm
!
!    nvtxs = N
!    allocate(edge(N))
!
!    do i = 1, N
!      in = index(i) - index(i-1)
!
!      if(0 < in)then
!        allocate(nozero(in-1))
!        nozero = 0
!
!        jn = 0
!        jS = index(i-1) + 1
!        jE = index(i  )
!        do j = jS, jE
!          if(item(j) /= i)then
!            jn = jn + 1
!            nozero(jn) = item(j)
!          endif
!        enddo
!
!        call reallocate_array(edge(i)%N, jn, edge(i)%node)
!        edge(i)%N = jn
!
!        do j = 1, jn
!          edge(i)%node(j) = nozero(j)
!        enddo
!        deallocate(nozero)
!      endif
!    enddo
!
!    allocate(xadj(N+1))
!    xadj(1) = 0
!    do i = 1, N
!      xadj(i+1) = xadj(i) + edge(i)%N
!    enddo
!
!    nedge = xadj(N+1)
!    allocate(adjncy(nedge))
!    in = 1
!    do i = 1, N
!      do j = 1, edge(i)%N
!        adjncy(in) = edge(i)%node(j) - 1
!        in = in + 1
!      enddo
!    enddo
!
!#ifdef WITH_METIS
!    call METIS_NodeND(nvtxs, xadj, adjncy, vwgt, options, perm, iperm)
!#endif
!
!    do i = 1, N
!       perm(i) =  perm(i) + 1
!      iperm(i) = iperm(i) + 1
!    enddo
!    do i = N+1, NP
!       perm(i) = i
!      iperm(i) = i
!    enddo
!
!    do i = 1, N
!      if(associated(edge(i)%node)) deallocate(edge(i)%node)
!    enddo
!    deallocate(edge)
!    deallocate(xadj)
!    deallocate(adjncy)
  end subroutine monolis_reorder_matrix_metis

  subroutine reallocate_array(in, inew, x)
    implicit none
    integer(kint), intent(in) :: in, inew
    integer(kint), pointer :: x(:), t(:)
    integer(kint) :: i

!    if(.not. associated(x))then
!      allocate(x(inew))
!    else
!      t => x
!      x => null()
!      allocate(x(inew))
!      do i=1,in
!        x(i) = t(i)
!      enddo
!      deallocate(t)
!    endif
  end subroutine reallocate_array


  subroutine monolis_spmat_restruct_comm(monoCOM, monoCOM_reorder, perm)
    implicit none
    type(monolis_com) :: monoCOM
    type(monolis_com) :: monoCOM_reorder
    integer(kint) :: perm(:)
    integer(kint) :: i, in, N

!    monoCOM_reorder%myrank   = monoCOM%myrank
!    monoCOM_reorder%comm     = monoCOM%comm
!    monoCOM_reorder%commsize = monoCOM%commsize
!    monoCOM_reorder%send_n_neib = monoCOM%send_n_neib
!    monoCOM_reorder%recv_n_neib = monoCOM%recv_n_neib
!
!    if(monoCOM%send_n_neib /= 0)then
!      monoCOM_reorder%recv_neib_pe => monoCOM%recv_neib_pe
!      monoCOM_reorder%recv_index   => monoCOM%recv_index
!      monoCOM_reorder%recv_item    => monoCOM%recv_item
!
!      monoCOM_reorder%send_neib_pe => monoCOM%send_neib_pe
!      monoCOM_reorder%send_index   => monoCOM%send_index
!
!      N = monoCOM%send_index(monoCOM%send_n_neib)
!      allocate(monoCOM_reorder%send_item(N))
!      do i = 1, N
!        in = perm(monoCOM%send_item(i))
!        monoCOM_reorder%send_item(i) = in
!      enddo
!    endif
  end subroutine monolis_spmat_restruct_comm

  subroutine monolis_spmat_restruct_matrix(monoMAT, monoMAT_reorder, perm, iperm)
    implicit none
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoMAT_reorder
    integer(kint) :: perm(:), iperm(:)
    integer(kint) :: N, NP, NZ, NDOF, NDOF2

!    N = monoMAT%N
!    NP = monoMAT%NP
!    NZ = monoMAT%index(NP)
!    NDOF = monoMAT%NDOF
!    NDOF2 = NDOF*NDOF
!
!    monoMAT_reorder%N = N
!    monoMAT_reorder%NP = NP
!    monoMAT_reorder%NZ = NZ
!    monoMAT_reorder%NDOF = NDOF
!    allocate(monoMAT_reorder%index(0:NP))
!    allocate(monoMAT_reorder%item(NZ))
!    call monolis_restruct_matrix_profile(NP, perm, iperm, &
!       & monoMAT%index, monoMAT%item, monoMAT_reorder%index, monoMAT_reorder%item)
!
!    allocate(monoMAT_reorder%A(NDOF2*NZ))
!    call monolis_restruct_matrix_values(NP, NDOF, perm, iperm, &
!       & monoMAT%index, monoMAT%item, monoMAT%A, &
!       & monoMAT_reorder%index, monoMAT_reorder%item, monoMAT_reorder%A)
!
!    allocate(monoMAT_reorder%X(NDOF*NP))
!    allocate(monoMAT_reorder%B(NDOF*NP))
  end subroutine monolis_spmat_restruct_matrix

  subroutine monolis_spmat_restruct_matrix_profile(N, perm, iperm, &
    & index, item, indexp, itemp)
    implicit none
    integer(kint) :: N
    integer(kint) :: perm(:), iperm(:)
    integer(kint) :: index(0:), item(:)
    integer(kint) :: indexp(0:), itemp(:)
    integer(kint) :: cnt, i, in, j, jo, jn

!    cnt = 0
!    indexp(0) = 0
!    do i = 1, N
!      in = perm(i)
!      do j = index(in-1)+1, index(in)
!        jo = item(j)
!        jn = iperm(jo)
!        cnt = cnt + 1
!        itemp(cnt) = jn
!      enddo
!      indexp(i) = cnt
!      call sort_int_array(itemp, indexp(i-1)+1, indexp(i))
!    enddo
  end subroutine monolis_spmat_restruct_matrix_profile

  subroutine monolis_spmat_restruct_matrix_values(N, NDOF, perm, iperm, index, item, A, &
      & indexp, itemp, Ap)
    implicit none
    integer(kint) :: N, NDOF
    integer(kint) :: perm(:), iperm(:)
    integer(kint) :: index(0:), item(:)
    real(kdouble) :: A(:)
    integer(kint) :: indexp(0:), itemp(:)
    real(kdouble) :: Ap(:)
    integer(kint) :: NDOF2, in, i
    integer(kint) :: jSn, jEn
    integer(kint) :: jo, ko, kn, jn, lo, ln, l

!    NDOF2 = NDOF*NDOF
!    do i = 1, N
!      in = iperm(i)
!      jSn = indexp(in-1)+1
!      jEn = indexp(in)
!      do jo = index(i-1)+1, index(i)
!        ko = item(jo)
!        kn = iperm(ko)
!        call bsearch_int_array(itemp, jSn, jEn, kn, jn)
!        lo = (jo-1)*NDOF2
!        ln = (jn-1)*NDOF2
!        do l = 1, NDOF2
!          Ap(ln + l) = A(lo + l)
!        enddo
!      enddo
!    enddo
  end subroutine monolis_spmat_restruct_matrix_values
end module mod_monolis_reorder