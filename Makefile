# MAKEFILE #

#INFORMANDO O COMPILADOR,
#DIRETÓRIOS E O
#NOME DO PROGRAMA
SRCCUDA=cuda/
SRCCOM=common/
SRCPAR=paralelo/
cuda=PPMcuda
common=common
LIBCU=-arch=sm_30 -Xptxas -dlcm=ca

SOURCES_CU=$(wildcard $(SRCCUDA)*.cu)
SOURCES=$(wildcard $(SRCCOM)*.c $(SRCPAR)*.c)

all: $(common) $(cuda)

$(cuda): $(SOURCES:.c=.o) $(SOURCES_CU:.cu=.o)
	nvcc -o $@ $^ $(LIBCU)


%.o: %.c
	nvcc -x cu -I. -dc $< -o $@

%.o: %.cu 
	nvcc -c $< -o $@ 

clean:
	rm -f $(SRCCUDA)*.o $(SRCCOM)*.o $(SRCPAR)*.o
	rm -f $(cuda)
