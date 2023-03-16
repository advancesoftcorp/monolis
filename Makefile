#> monolis_utils Makefile

##> compiler setting
FC     = mpif90
FFLAGS = -fPIC -O2 -mtune=native -march=native -std=legacy -Wno-missing-include-dirs
CC     = mpicc
CFLAGS = -fPIC -O2

##> directory setting
MOD_DIR = -J ./include
INCLUDE = -I /usr/include -I ./include -I ./submodule/gedatsu/include -I ./submodule/monolis_utils/include
USE_LIB = \
-L./lib -lmonolis \
-L./submodule/gedatsu/lib -lgedatsu \
-L./submodule/monolis_utils/lib -lmonolis_utils \
-L./lib -lmetis \
-llapack -lblas
BIN_DIR = ./bin
SRC_DIR = ./src
OBJ_DIR = ./obj
LIB_DIR = ./lib
WRAP_DIR= ./wrapper
TST_DIR = ./test
DRV_DIR = ./driver
LIBRARY = libmonolis.a
CPP     = -cpp $(FLAG_DEBUG)

##> option setting
ifdef FLAGS
	comma:= ,
	empty:=
	space:= $(empty) $(empty)
	DFLAGS = $(subst $(comma), $(space), $(FLAGS))

	ifeq ($(findstring DEBUG, $(DFLAGS)), DEBUG)
		FFLAGS  = -fPIC -O2 -std=legacy -fbounds-check -fbacktrace -Wuninitialized -ffpe-trap=invalid,zero,overflow -Wno-missing-include-dirs
	endif

	ifeq ($(findstring INTEL, $(DFLAGS)), INTEL)
		FC      = mpiifort
		FFLAGS  = -fPIC -O2 -align array64byte
		CC      = mpiicc
		CFLAGS  = -fPIC -O2 -no-multibyte-chars
		MOD_DIR = -module ./include
	endif
endif

##> other commands
MAKE = make
CD   = cd
CP   = cp
RM   = rm -rf
AR   = - ar ruv

##> **********
##> target (1)
LIB_TARGET = $(LIB_DIR)/$(LIBRARY)

##> source file define
SRC_DEFINE = \
def_solver.f90 \
def_mat.f90 \
def_struc.f90 \
def_solver_util.f90

SRC_MAT = \
spmat_handler_util.f90 \
spmat_nzpattern_util.f90 \
spmat_nzpattern.f90 \
spmat_handler.f90

#spmat_fillin.f90 \
#spmat_copy.f90 \
#spmat_reorder.f90 \
#spmat_scaling.f90 \

SRC_LINALG = \
matvec.f90 \
inner_product.f90 \
mat_converge.f90 \
vec_util.f90

SRC_WRAP = \
wrapper_lapack.f90

#matmat.f90 \

#SRC_FACT = \
#11/fact_LU_11.f90 \
#11/fact_MF_11.f90 \
#33/fact_LU_33.f90 \
#nn/fact_LU_nn.f90 \
#fact_LU.f90 \
#fact_MF.f90

SRC_PREC = \
33/diag_33.f90 \
33/sor_33.f90 \
nn/diag_nn.f90 \
nn/sor_nn.f90 \
diag.f90 \
sor.f90 \
precond.f90

#MUMPS.f90 \
#ilu.f90 \
#Jacobi.f90 \
#MF.f90 \

SRC_ITER = \
CG.f90 \
BiCGSTAB.f90 \
BiCGSTAB_noprec.f90 \
GropCG.f90 \
PipeCG.f90 \
PipeCR.f90 \
PipeBiCGSTAB.f90 \
PipeBiCGSTAB_noprec.f90 \
COCG.f90

#CABiCGSTAB_noprec.f90 \
#GMRES.f90 \

SRC_SOLV = \
solver.f90

SRC_EIGEN = \
Lanczos_util.f90 \
Lanczos.f90 \
eigen_solver.f90

##> C wrapper section
SRC_DEFINE_C = \
monolis_def_solver_c.c \
monolis_def_mat_c.c \
monolis_def_struc_c.c \
monolis_def_solver_util_c.c

SRC_LINALG_C = \
matvec_wrap.f90 \
inner_product_wrap.f90 \
monolis_matvec_c.c \
monolis_inner_product_c.c

SRC_MAT_C = \
monolis_spmat_nzpattern_util_c.c\
monolis_spmat_nzpattern_c.c \
monolis_spmat_handler_c.c \
spmat_nzpattern_util_wrap.f90 \
spmat_handler_util_wrap.f90

SRC_ALL_C = \
$(addprefix define/, $(SRC_DEFINE_C)) \
$(addprefix linalg/, $(SRC_LINALG_C)) \
$(addprefix matrix/, $(SRC_MAT_C))

##> all targes
SRC_ALL = \
$(addprefix define/, $(SRC_DEFINE)) \
$(addprefix matrix/, $(SRC_MAT)) \
$(addprefix linalg/, $(SRC_LINALG)) \
$(addprefix wrapper/, $(SRC_WRAP)) \
$(addprefix fact/, $(SRC_FACT)) \
$(addprefix prec/, $(SRC_PREC)) \
$(addprefix iterative/, $(SRC_ITER)) \
$(addprefix solver/, $(SRC_SOLV)) \
$(addprefix eigen/, $(SRC_EIGEN)) \

##> lib objs
LIB_SOURCES = \
$(addprefix $(SRC_DIR)/,  $(SRC_ALL)) \
$(addprefix $(WRAP_DIR)/, $(SRC_ALL_C)) \
./src/monolis.f90
LIB_OBJSt   = $(subst $(SRC_DIR), $(OBJ_DIR), $(LIB_SOURCES:.f90=.o))
LIB_OBJS    = $(subst $(WRAP_DIR), $(OBJ_DIR), $(LIB_OBJSt:.c=.o))

##> **********
##> test target (2)
TEST_TARGET = $(TST_DIR)/monolis_test

##> lib objs
TST_SRC_ALL = $(SRC_ALL) monolis.f90
TST_SOURCES = $(addprefix $(TST_DIR)/, $(TST_SRC_ALL))
TST_OBJSt   = $(subst $(TST_DIR), $(OBJ_DIR), $(TST_SOURCES:.f90=_test.o))
TST_OBJS    = $(TST_OBJSt:.c=_test.o)

##> target
all: \
	cp_header \
	$(LIB_TARGET) \
	$(TEST_TARGET)

lib: \
	cp_header \
	$(LIB_TARGET)

$(LIB_TARGET): $(LIB_OBJS)
	$(AR) $@ $(LIB_OBJS) $(ARC_LIB)

$(TEST_TARGET): $(TST_OBJS)
	$(FC) $(FFLAGS) $(CPP) $(INCLUDE) -o $@ $(TST_OBJS) $(USE_LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	$(FC) $(FFLAGS) $(CPP) $(INCLUDE) $(MOD_DIR) -o $@ -c $<

$(OBJ_DIR)/%.o: $(TST_DIR)/%.f90
	$(FC) $(FFLAGS) $(CPP) $(INCLUDE) $(MOD_DIR) -o $@ -c $<

$(OBJ_DIR)/%.o: $(WRAP_DIR)/%.f90
	$(FC) $(FFLAGS) $(CPP) $(INCLUDE) $(MOD_DIR) -o $@ -c $<

$(OBJ_DIR)/%.o: $(WRAP_DIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

cp_header:
	$(CP) ./wrapper/linalg/monolis_matvec_c.h ./include/
	$(CP) ./wrapper/linalg/monolis_inner_product_c.h ./include/
	$(CP) ./wrapper/define/monolis_def_struc_c.h ./include/
	$(CP) ./wrapper/define/monolis_def_mat_c.h ./include/
	$(CP) ./wrapper/define/monolis_def_solver_util_c.h ./include/
	$(CP) ./wrapper/define/monolis_def_solver_c.h ./include/
	$(CP) ./wrapper/matrix/monolis_spmat_nzpattern_c.h ./include/
	$(CP) ./wrapper/matrix/monolis_spmat_nzpattern_util_c.h ./include/
	$(CP) ./wrapper/monolis.h ./include/

clean:
	$(RM) \
	$(LIB_OBJS) \
	$(TST_OBJS) \
	$(LIB_TARGET) \
	$(TEST_TARGET) \
	./include/*.mod \
	./bin/*

.PHONY: clean
