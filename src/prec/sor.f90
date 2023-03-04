!> SOR 前処理関連モジュール
module mod_monolis_precond_sor
  use mod_monolis_utils
  use mod_monolis_def_mat
  use mod_monolis_def_struc
  use mod_monolis_precond_sor_33
  use mod_monolis_precond_sor_nn

  implicit none

contains

  !> 前処理生成：SOR 前処理
  subroutine monolis_precond_sor_setup_R(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_sor_setup_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_sor_33_setup_R(monoMAT, monoPREC)
    else
      call monolis_precond_sor_nn_setup_R(monoMAT, monoPREC)
    endif
  end subroutine monolis_precond_sor_setup_R

  !> 前処理適用：SOR 前処理
  subroutine monolis_precond_sor_apply_R(monoPRM, monoCOM, monoMAT, monoPREC, X, Y)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC
    real(kdouble) :: X(:), Y(:)

    call monolis_std_debug_log_header("monolis_precond_sor_apply_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_sor_33_apply_R(monoMAT, monoPREC, X, Y)
    else
      call monolis_precond_sor_nn_apply_R(monoMAT, monoPREC, X, Y)
    endif
  end subroutine monolis_precond_sor_apply_R

  !> 前処理初期化：SOR 前処理
  subroutine monolis_precond_sor_clear_R(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_sor_clear_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_sor_33_clear_R(monoPREC)
    else
      call monolis_precond_sor_nn_clear_R(monoPREC)
    endif
  end subroutine monolis_precond_sor_clear_R
end module mod_monolis_precond_sor