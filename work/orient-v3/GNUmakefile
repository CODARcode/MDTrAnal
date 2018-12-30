PROG = orient.exe

OFILES = \
orient.o 

LIB=-L/home/hvandam/lib

all:	$(PROG)

$(PROG): $(OFILES)
	gfortran -g -o $(PROG) $(OFILES) $(LIB) -llapack -lblas

.F.o:
	gfortran -g -c $*.F
