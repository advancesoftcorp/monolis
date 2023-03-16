!> 疎行列操作関数群（メイン関数）
module mod_monolis_spmat_handler_util_wrap
  use mod_monolis_utils
  use mod_monolis_spmat_handler_util
  use iso_c_binding

  implicit none

contains

  subroutine monolis_add_sparse_matrix_c(N, NZ, NDOF, NBF, index, item, A, conn, mat) &
    & bind(c, name = "monolis_add_sparse_matrix_c_main")
    implicit none
    integer(c_int), intent(in), value :: N, NZ, NDOF, NBF
    integer(c_int), intent(in), target :: index(0:N)
    integer(c_int), intent(in), target :: item(NZ)
    integer(c_int), intent(in), target :: conn(NBF)
    real(c_double), target :: A(NDOF*NDOF*NZ)
    real(c_double), target :: mat(NDOF*NDOF*NBF*NBF)
    integer(kint) :: conn_t(NBF)

    conn_t = conn + 1
    call monolis_sparse_matrix_add_matrix(index, item, A, NBF, NBF, NDOF, conn_t, conn_t, mat)
  end subroutine monolis_add_sparse_matrix_c

  subroutine monolis_get_scalar_from_sparse_matrix_c(N, NZ, NDOF, index, item, A, i, j, sub_i, sub_j, val, is_find) &
    & bind(c, name = "monolis_get_scalar_from_sparse_matrix_c_main")
    implicit none
    integer(c_int), intent(in), value :: N, NZ, NDOF, i, j, sub_i, sub_j
    integer(c_int), intent(in), target :: index(0:N)
    integer(c_int), intent(in), target :: item(NZ)
    integer(c_int), target :: is_find
    real(c_double), target :: A(NDOF*NDOF*NZ)
    real(c_double), target :: val
    integer(kint) :: i_t, j_t, sub_i_t, sub_j_t
    logical :: is_find_t

    i_t = i + 1
    j_t = j + 1
    sub_i_t = sub_i + 1
    sub_j_t = sub_j + 1
    call monolis_sparse_matrix_get_value(index, item, A, NDOF, i_t, j_t, sub_i_t, sub_j_t, val, is_find_t)
    is_find = 0
    if(is_find_t) is_find = 1
  end subroutine monolis_get_scalar_from_sparse_matrix_c

  subroutine monolis_set_Dirichlet_bc_c(N, NZ, NDOF, index, item, indexR, itemR, permR, &
    & A, B, nid, ndof_bc, val) &
    & bind(c, name = "monolis_set_Dirichlet_bc_c_main")
    implicit none
    integer(c_int), intent(in), value :: N, NZ, NDOF, nid, ndof_bc
    integer(c_int), intent(in), target :: index(0:N)
    integer(c_int), intent(in), target :: item(NZ)
    integer(c_int), intent(in), target :: indexR(0:N)
    integer(c_int), intent(in), target :: itemR(NZ)
    integer(c_int), intent(in), target :: permR(NZ)
    real(c_double), intent(in), value :: val
    real(c_double), target :: A(NDOF*NDOF*NZ)
    real(c_double), target :: B(NDOF*N)
    integer(kint) :: nid_t, ndof_bc_t

    nid_t = nid + 1
    ndof_bc_t = ndof_bc + 1
    call monolis_sparse_matrix_add_bc(index, item, A, B, indexR, itemR, permR, &
      & ndof, nid_t, ndof_bc_t, val)
  end subroutine monolis_set_Dirichlet_bc_c

end module mod_monolis_spmat_handler_util_wrap
