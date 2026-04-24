#!/bin/bash

export LIB="-lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -fopenmp -nofor-main"

mpif90 -I../../include \
-std=legacy -fbounds-check -fbacktrace -Wuninitialized -ffpe-trap=invalid,zero,overflow \
-o mesher mesher.f90 \
-L../../lib -lmonolis_solver -lgedatsu -lmonolis_utils -lmetis ${LIB}

mpirun -np 1 ./mesher -i mtx.dat

mpirun -np 1 ../../bin/gedatsu_simple_mesh_partitioner -n 3

mpif90 -I../../include \
-std=legacy -fbounds-check -fbacktrace -Wuninitialized -ffpe-trap=invalid,zero,overflow \
-o solver main.f90 \
-L../../lib -lmonolis_solver -lgedatsu -lmonolis_utils -lmetis ${LIB}

mpirun -np 1 ./solver

mpirun -np 3 ./solver

