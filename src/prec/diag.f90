!> 対角スケーリング前処理関連モジュール
module mod_monolis_precond_diag
  use mod_monolis_utils
  use mod_monolis_def_mat
  use mod_monolis_def_struc
  use mod_monolis_precond_diag_33
  use mod_monolis_precond_diag_nn

  implicit none

contains

  !> 前処理生成：対角スケーリング前処理（実数型）
  subroutine monolis_precond_diag_setup_R(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_diag_setup_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_diag_33_setup_R(monoMAT, monoPREC)
    else
      call monolis_precond_diag_nn_setup_R(monoMAT, monoPREC)
    endif
  end subroutine monolis_precond_diag_setup_R

  !> 前処理生成：対角スケーリング前処理（複素数型）
  subroutine monolis_precond_diag_setup_C(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_diag_setup_C")

    call monolis_precond_diag_nn_setup_C(monoMAT, monoPREC)
  end subroutine monolis_precond_diag_setup_C

  !> 前処理適用：対角スケーリング前処理（実数型）
  subroutine monolis_precond_diag_apply_R(monoPRM, monoCOM, monoMAT, monoPREC, X, Y)
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

    call monolis_std_debug_log_header("monolis_precond_diag_apply_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_diag_33_apply_R(monoMAT, monoPREC, X, Y)
    else
      call monolis_precond_diag_nn_apply_R(monoMAT, monoPREC, X, Y)
    endif
  end subroutine monolis_precond_diag_apply_R

  !> 前処理適用：対角スケーリング前処理（複素数型）
  subroutine monolis_precond_diag_apply_C(monoPRM, monoCOM, monoMAT, monoPREC, X, Y)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC
    complex(kdouble) :: X(:), Y(:)

    call monolis_std_debug_log_header("monolis_precond_diag_apply_C")

    call monolis_precond_diag_nn_apply_C(monoMAT, monoPREC, X, Y)
  end subroutine monolis_precond_diag_apply_C

  !> 前処理初期化：対角スケーリング前処理（実数型）
  subroutine monolis_precond_diag_clear_R(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_diag_clear_R")

    if(monoMAT%NDOF == 3)then
      call monolis_precond_diag_33_clear_R(monoPREC)
    else
      call monolis_precond_diag_nn_clear_R(monoPREC)
    endif
  end subroutine monolis_precond_diag_clear_R

  !> 前処理初期化：対角スケーリング前処理（複素数型）
  subroutine monolis_precond_diag_clear_C(monoPRM, monoCOM, monoMAT, monoPREC)
    implicit none
    !> パラメータ構造体
    type(monolis_prm) :: monoPRM
    !> 通信テーブル構造体
    type(monolis_com) :: monoCOM
    !> 行列構造体
    type(monolis_mat) :: monoMAT
    !> 前処理構造体
    type(monolis_mat) :: monoPREC

    call monolis_std_debug_log_header("monolis_precond_diag_clear_C")

    call monolis_precond_diag_nn_clear_C(monoPREC)
  end subroutine monolis_precond_diag_clear_C
end module mod_monolis_precond_diag