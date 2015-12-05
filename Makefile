# MAKEFILE #

#INFORMANDO O COMPILADOR,
#DIRETÓRIOS E O
#NOME DO PROGRAMA
CC=gcc
G=g++
MPICC=mpicc
MPIG=mpic++
NVCC=nvcc
SRCCOMMON=common/
SRCSEQ=sequencial/
SRCMPI=mpi/
SRCCUD=cuda/
SEQ=PPMseq
MPI=PPMmpi
CUDA=PPMcuda

# FLAGS NECESSARIAS
# PARA COMPILACAO
CFLAGS=-Wall -Wextra -fopenmp -lmpi
LIBCU=-arch=sm_30 -Xptxas -dlcm=ca

#-------------------------------
# CARREGA AUTOMATICAMENTE OS
# ARQUIVOS .C E .H
#-------------------------------
SOURCESEQ=$(wildcard $(SRCSEQ)*.c $(SRCCOMMON)*.c)
HEADERSEQ=$(wildcard $(SRCSEQ)*.h $(SRCCOMMON)*.h)

SOURCEMPI=$(wildcard $(SRCMPI)*.c $(SRCCOMMON)*.c)
HEADERMPI=$(wildcard $(SRCMPI)*.h $(SRCCOMMON)*.h)

SOURCECOM=$(wildcard $(SRCCOMMON)*.c)

all: $(SEQ) $(MPI) $(CUDA)

################################## M P I ######################################
$(MPI): $(SOURCEMPI:.c=.o)
	$(MPIG) -o $@ $^ $(CFLAGS)

%.o: %.c $(HEADERMPI)
	$(MPICC) -g -c $< -o $@ $(CFLAGS)

############################### S E Q U E N C I A L ###########################
$(SEQ): $(SOURCESEQ:.c=.o)
	$(G) -o $@ $^ $(CFLAGS)

%.o: %.c $(HEADERSEQ)
	$(CC) -g -c $< -o $@ $(CFLAGS)

####################################### C U D A ###############################
$(CUDA): cuda/main.o cuda/menu.o cuda/cuda.o cuda/imagem.o cuda/funcao.o cuda/timer.o
	$(NVCC) -o $@ $^ $(LIBCU)

%.o: %.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/imagem.o: common/imagem.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/funcao.o: common/funcao.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/timer.o: common/timer.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/main.o: cuda/main.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/menu.o: cuda/menu.c
	$(NVCC) -x cu -I. -dc $< -o $@

cuda/cuda.o: cuda/cuda.cu
	$(NVCC) -c $< -o $@
####################################### C U D A ###############################

clean:
	rm -f $(SRCSEQ)*.o $(SRCMPI)*.o $(SRCCOMMON)*.o $(SRCCUD)*.o
	rm -f $(SEQ) $(MPI) $(CUDA)

