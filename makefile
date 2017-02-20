DIR = ./
PROGRAMS = $(DIR)main
CC = mpic++     # Warning Must be the same as in Library path
#CC = g++
DIR       = OBJECTS/
DIREX     = RUN/


CFLAGS=-O3 -std=c++0x
LINKER= $(CC) $(CFLAGS)
LIBS=  -L/user/local/lib -lmpi -lfftw3 -lmpl -lpthread -L/home/igor/Programs/lammps-30Jul16/src -llammps_mpi

INCLUDES=  -I/user/local/include -I/home/igor/Programs/lammps-30Jul16/src

# all: $(PROGRAMS)
# .cpp: ;  $(CC) $(CFLAGS) $(INCLUDES) $@.cpp $(LIBS) -o ../m3d

OBJ = 	$(DIR)main.o\
		

MAKEFILE = makefile

$(DIR)%.o: %.cpp $(MAKEFILE)
	$(CC) -c $(CFLAGS) -o $@ $(INCLUDES) $*.cpp

$(DIREX)noname:  $(OBJ) $(MAKEFILE)
	$(LINKER) $(OBJ)  -o $@ $(LIBS)

clean:  
	rm -f $(DIR)*.o  
	rm -f $(DIREX)history.dat
	rm -f $(DIREX)BOX.dump    
	rm -f $(DIREX)Data/data.init.*
	rm -f $(DIREX)test.txt
	rm -f $(DIREX)stress.dat
	#rm -r $(DIREX)tempLAMMPS_*
