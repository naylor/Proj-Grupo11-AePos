# MAKEFILE #

#INFORMANDO O COMPILADOR,
#DIRETÓRIOS E O
#NOME DO PROGRAMA
SRCCUDA=cuda/
SRCCOM=common/
SRCPAR=paralelo/
cuda=PPMcuda
LIB=-arch=sm_35

SOURCES=$(wildcard $(SRCPAR)*.cu $(SRCCOM)*.cu $(SRCCUDA)*.cu)

all: $(cuda)

$(cuda): $(SOURCES:.cu=.o)
	nvcc -o $@ $^ $(LIB)

%.o: %.cu 
	nvcc -c $< -o $@ $(LIB)

clean:
	rm -f $(SRCCUDA)*.o $(SRCCOM)*.o $(SRCPAR)*.o
	rm -f $(cuda)
