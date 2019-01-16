
#requirespackage   'PETSCOPT_HAVE_MFEMOPT'

ALL: lib

CFLAGS    =
FFLAGS    =
SOURCECPP = coefficients.cpp  nonlininteg.cpp  objective.cpp  parametermap.cpp  pdbilininteg.cpp  pdcoefficient.cpp  pdoperator.cpp  petsctaosolver.cpp  receiver.cpp  reducedfunctional.cpp
SOURCEF   =
SOURCEH   =
DIRS      = examples
LIBBASE   = libpetscopt
MANSEC    = TS
LOCDIR    = src/mfemopt/

include ${PETSCOPT_DIR}/lib/petscopt/conf/variables
include ${PETSCOPT_DIR}/lib/petscopt/conf/rules
include ${PETSCOPT_DIR}/lib/petscopt/conf/test
