Usando CUDA para aplicação de Smooth em Imagens PPM/PGM
===========================================================

### Dependências
Esse programa utiliza CUDA para aplicar Smooth (filtro de média 5x5) em imagens PPM e PGM.

### Dependências
1. Interface Gráfica NVIDIA

2. Pacotes necessários:

* sudo apt-get install nvidia-cuda-toolkit

### Instalação

1. Faça o clone deste projeto:
	git clone https://github.com/naylor/Proj-Grupo11-AePos

2. Entre na pasta do projeto

3. Rode o comando "make"


### Executando a aplicação
2. Executando o PPMcuda pelo terminal
   usar: ./PPMcuda --help
   ou
   usar: Usar: ./PPMcuda -i [IMAGEM] -a [MEMORIA ASSINCRONA] -s [MEMORIA COMPARTILHADA] -c [CARGA TRABALHO(Opcional)] -d [NIVEL DEBUG(Opcional)]


  * [IMAGEM]: colocar apenas o nome do arquivo (ex. model.ppm, omitir o diretório).
  * [MEMÓRIA ASSÍNCRONA]: se ativado, cudaMemcpyAsync é utilizado para cópia da imagem.
  * [MEMÓRIA COMPARTILHADA]: se ativado, a Shared Memory é utilizada em blocos de 32x32.
  * [CARGA TRABALHO]: número máximo de linhas, que o Rank0 alocará para cada processo, se omitido, será uma divisão igualitária.
  * [CARGA ALEATÓRIA]: se ativado, as cargas enviadas para os nodes serão aleatórias.
  * [LEITURA INDIVIDUAL]: faz com que cada processo tenha acesso exclusivo a imagem no momento da leitura.
  * [NIVEL DEBUG]: permite monitorar os eventos do sistema.

3. Os resultados são gravados na pasta: resultados

4. Imagens PPM/PGM disponíveis na pasta: images_in
5. Imagens processadas com Smooth na pasta: images_out
