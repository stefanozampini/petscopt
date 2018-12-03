
CFLAGS          =
FFLAGS          =
CPPFLAGS        =
FPPFLAGS        =
LOCDIR          = src/tao/examples/tests/petsc
EXAMPLESC       = burgers_spectral.c spectraladjointassimilation.c
EXAMPLESF       =
MANSEC          = Tao
DIRS            =
CLEANFILES      =

include ${PETSCOPT_DIR}/lib/petscopt/conf/variables
include ${PETSCOPT_DIR}/lib/petscopt/conf/rules
include ${PETSCOPT_DIR}/lib/petscopt/conf/test

spectraladjointassimilation: spectraladjointassimilation.o
	-${CLINKER} -o spectraladjointassimilation spectraladjointassimilation.o ${PETSC_LIB}
	${RM} spectraladjointassimilation.o

burgers_spectral: burgers_spectral.o
	-${CLINKER} -o burgers_spectral burgers_spectral.o ${PETSC_LIB}
	${RM} burgers_spectral.o

