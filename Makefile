# MAKEFILE #

#INFORMANDO O COMPILADOR,
#DIRETÓRIOS E O
#NOME DO PROGRAMA
SRCCUDA=cuda/
SRCCOM=common/
SRCPAR=paralelo/
cuda=PPMcuda
LIB=-L/usr/lib/x86_64-linux-gnu/libcudart.so -lcudart

SOURCES=$(wildcard $(SRCPAR)*.cu $(SRCCOM)*.cu $(SRCCUDA)*.cu)

all: $(cuda)

$(cuda): $(SOURCES:.cu=.o)
	nvcc -o $@ $^

%.o: %.cu 
	nvcc -c $< -o $@ 

clean:
	rm -f $(SRCCUDA)*.o $(SRCCOM)*.o $(SRCPAR)*.o
	rm -f $(cuda)
