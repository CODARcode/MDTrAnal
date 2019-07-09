ADIOS2_DIR=/opt/adios2

# mpif77
FC=mpif90
FFLAGS=-g -O2

override ADIOS2_FINC=`${ADIOS2_DIR}/bin/adios2-config --fortran-flags`
override ADIOS2_FLIB=`${ADIOS2_DIR}/bin/adios2-config --fortran-libs`

all: adios2

sp_io.o: sp_io.f
	${FC} ${FFLAGS} -c ${ADIOS2_FINC} $<

sp_io_vars.o: sp_io_vars.f
	${FC} ${FFLAGS} -c ${ADIOS2_FINC} $<

adios2: sp_io_vars.o sp_io.o sp_io_test.f90
	${FC} ${FFLAGS} -o sp_io_test ${ADIOS2_FINC} $^ ${ADIOS2_FLIB}

clean:
	rm -f *.o *.mod sp_io_test
	rm -rf *.bp*