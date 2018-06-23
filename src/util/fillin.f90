module mod_monolis_matrix_fillin
  use mod_monolis_prm
  use mod_monolis_com
  use mod_monolis_mat

  implicit none

  type monolis_fillin
    integer(kind=kint) :: n_descendant
    integer(kind=kint) :: n_ancestor
    integer(kind=kint), pointer :: descendant(:)
    integer(kind=kint), pointer :: ancestor(:)
    integer(kind=kint), pointer :: update_index(:)
    real(kind=kdouble), pointer :: update(:,:,:)
    logical :: factorized
    logical :: updated
  endtype monolis_fillin

contains

  subroutine monolis_matrix_get_fillin(monoPRM, monoCOM, monoMAT, monoTREE, is_fillin, is_asym)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoTREE
    type(monolis_fillin), pointer:: tree(:)
    integer(kind=kint), pointer :: idxU(:)
    integer(kind=kint), pointer :: itemU(:)
    integer(kind=kint), pointer :: idxL(:)
    integer(kind=kint), pointer :: itemL(:)
    integer(kind=kint), pointer :: array(:)
    integer(kind=kint), pointer :: fillin_mask(:)
    integer(kind=kint), pointer :: child_mask(:)
    integer(kind=kint), pointer :: parent_mask(:)
    integer(kind=kint) :: N, NPU, NPL
    integer(kind=kint) :: i, j, k, jS, jE, in, c
    integer(kind=kint) :: Nbytes
    integer(kind=kint) :: is, ie
    integer(kind=kint) :: range, parent
    integer(kind=kint) :: bit = kint*8
    integer(kind=kint), allocatable :: count(:), diff(:)
    logical :: is_fillin, is_asym

    N = monoMAT%N
    allocate(tree(N))

    do i = 1, N
      tree(i)%factorized = .false.
      tree(i)%updated = .false.
      tree(i)%n_descendant = 0
      tree(i)%n_ancestor = 0

      jS = monoMAT%indexU(i-1) + 1
      jE = monoMAT%indexU(i  )
      in = 0
      do j = jS, jE
        if(monoMAT%itemU(j) <= N)then
          in = in + 1
        endif
      enddo
      tree(i)%n_ancestor = in
      allocate(tree(i)%ancestor(in))

      in = 0
      do j = jS, jE
        if(monoMAT%itemU(j) <= N)then
          in = in + 1
          tree(i)%ancestor(in) = monoMAT%itemU(j)
        endif
      enddo
    enddo

    if(is_fillin)then
      Nbytes = N/bit+1
      allocate(child_mask (Nbytes))
      allocate(parent_mask(Nbytes))
      allocate(fillin_mask(Nbytes))

      do i = 1, N
        if(tree(i)%n_ancestor < 2) cycle
        is = i/bit + 1
        child_mask(is:Nbytes) = 0
        parent_mask(is:Nbytes) = 0

        parent = tree(i)%ancestor(1)
        range = 0
        do j = 2, tree(i)%n_ancestor
          in = tree(i)%ancestor(j)
          ie = in/bit + 1
          child_mask(ie) = ibset(child_mask(ie),mod(in,bit))
          range = in
        enddo
        k = tree(parent)%n_ancestor
        do j = 1, k
          in = tree(parent)%ancestor(j)
          ie = in/bit + 1
          parent_mask(ie) = ibset(parent_mask(ie),mod(in,bit))
          range = max(range,in)
        enddo
        ie = range/bit + 1

        fillin_mask(is:ie) = ior(child_mask(is:ie), parent_mask(is:ie))

        c = 0
        do j = is, ie
          c = c + popcnt(fillin_mask(j))
        enddo

        if(0 < c)then
          allocate(array(c))
          tree(parent)%n_ancestor=c
          in = 0
          do j = is, ie
            do k = 1, popcnt(fillin_mask(j))
              in = in + 1
              c = popcnt( iand(fillin_mask(j), - fillin_mask(j)) -1 )
              fillin_mask(j) = ibclr(fillin_mask(j),c)
              array(in) = bit*(j-1)+c
            enddo
          enddo
          deallocate(tree(parent)%ancestor)
          tree(parent)%ancestor => array
        endif
      enddo
      deallocate(child_mask )
      deallocate(parent_mask)
      deallocate(fillin_mask)

      !do i = 1, N
      !  do j = 1, tree(i)%n_ancestor
      !    tree(tree(i)%ancestor(j))%n_descendant = tree(tree(i)%ancestor(j))%n_descendant + 1
      !  enddo
      !enddo
      !do i = 1, N
      !  allocate(tree(i)%descendant(tree(i)%n_descendant))
      !enddo

      !allocate(diff(N))
      !diff(:) = 0
      !do i = 1, N
      !  do j = 1, tree(i)%n_ancestor
      !    diff(tree(i)%ancestor(j)) = diff(tree(i)%ancestor(j)) + 1
      !    tree(tree(i)%ancestor(j))%descendant(diff(tree(i)%ancestor(j))) = i
      !  enddo
      !enddo
    endif

    allocate(monoTREE%indexU(0:N))
    idxU => monoTREE%indexU
    in = 0
    idxU(0) = 0
    do i = 1, N
      idxU(i) = idxU(i-1) + tree(i)%n_ancestor + 1
      in = in + tree(i)%n_ancestor + 1
    enddo
    NPU = in
    monoTREE%NPU = in

    allocate(monoTREE%itemU(NPU))
    itemU => monoTREE%itemU
    in = 0
    do i = 1, N
      in = in + 1
      itemU(in) = i
      do j = 1, tree(i)%n_ancestor
        in = in + 1
        itemU(in) = tree(i)%ancestor(j)
      enddo
    enddo

    if(is_asym)then
      !lower part
      idxL => monoTREE%indexL
      itemL => monoTREE%itemL

      allocate(count(N))
      allocate(idxL(0:N))
      count = 0
      idxL(0) = 0

      do i = 1, idxU(N)
        in = itemU(i)
        count(in) = count(in) + 1
      enddo

      do i = 1, N
        idxL(i) = idxL(i-1) + count(i)
      enddo

      NPL = NPU
      c = 1
      allocate(itemL(NPL))
      do i = 1, N
        aa:do k = 1, i
          jS = idxU(k-1) + 1
          jE = idxU(k)
          do j = jS, jE
            in = itemU(j)
            if(i < in) cycle aa
            if(i == in)then
              itemL(c) = k
              c = c + 1
              cycle aa
            endif
          enddo
        enddo aa
      enddo
      deallocate(count)
    endif
  end subroutine monolis_matrix_get_fillin

  subroutine monolis_matrix_copy_with_fillin(monoPRM, monoCOM, monoMAT, monoTREE, is_asym)
    implicit none
    type(monolis_prm) :: monoPRM
    type(monolis_com) :: monoCOM
    type(monolis_mat) :: monoMAT
    type(monolis_mat) :: monoTREE
    integer(kind=kint), pointer :: idxU(:)
    integer(kind=kint), pointer :: itemU(:)
    integer(kind=kint), pointer :: idxL(:)
    integer(kind=kint), pointer :: itemL(:)
    real(kind=kdouble), pointer :: AU(:)
    real(kind=kdouble), pointer :: AL(:)
    integer(kind=kint) :: N, NPU, NPL, NDOF, NDOF2
    integer(kind=kint) :: i, j, k, l, iS, iE, jS, jE, lS, lE
    integer(kind=kint) :: in, jn ,kn, nn
    logical :: is_asym

    N = monoTREE%N
    NPU = monoTREE%NPU
    NDOF = monoTREE%NDOF
    NDOF2 = NDOF*NDOF
    idxU => monoTREE%indexU
    itemU => monoTREE%itemU
    idxL => monoTREE%indexL
    itemL => monoTREE%itemL

    !value
    allocate(monoTREE%AU(NDOF2*NPU))
    AU => monoTREE%AU
    AU = 0.0d0

    do k = 1, N
      in = idxU(k-1)+1
      jS = NDOF2*(in-1)
      jE = NDOF2*(k -1)
      do j = 1, NDOF2
        AU(jS + j) = monoMAT%D(jE + j)
      enddo
    enddo

    do k = 1, N
      iS = idxU(k-1) + 1
      iE = idxU(k)
      jS = monoMAT%indexU(k-1) + 1
      jE = monoMAT%indexU(k)
      aa:do j = jS, jE
        jn = monoMAT%itemU(j)
        do i = iS, iE
          in = itemU(i)
          if(jn == in)then
            lS = NDOF2*(i-1)
            lE = NDOF2*(j-1)
            do l = 1, NDOF2
              AU(lS + l) = monoMAT%AU(lE + l)
            enddo
            iS = iS + 1
            cycle aa
          endif
        enddo
      enddo aa
    enddo

    if(is_asym)then
      allocate(AL(9*NPL))
      AL = 0.0d0

      do k = 1, N
        iS = idxL(k-1) + 1
        iE = idxL(k)
        jS = monoMAT%indexL(k-1) + 1
        jE = monoMAT%indexL(k)
        bb:do j = jS, jE
          jn = monoMAT%itemL(j)
          do i = iS, iE
            in = itemL(i)
            if(jn == in)then
              lS = NDOF2*(i-1)
              lE = NDOF2*(j-1)
              do l = 1, NDOF2
                AL(lS + l) = monoMAT%AL(lE + l)
              enddo
              iS = iS + 1
              cycle bb
            endif
          enddo
        enddo bb
      enddo
    endif
  end subroutine monolis_matrix_copy_with_fillin
end module mod_monolis_matrix_fillin