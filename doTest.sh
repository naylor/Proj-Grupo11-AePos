#! /bin/bash

echo -e "Iniciando as coletas de dados - MPI";

for i in `ls images_in/`
  do
  for h in nodes_6 nodes_12
    do
    for r in {1,0}
      do
      for t in {1,5,8}
        do
        for n in {6,11,16,21}
          do
          for j in $(seq 10);
            do
            mpirun -n $n -hostfile $h ./PPMmpi -i $i -t $t -r $r -d 2
            sleep 1
          done
        done
      done
    done
  done
done

echo -e "Iniciando as coletas de dados - SEQUENCIAL";

for i in `ls images_in/`
  do
  for j in $(seq 10);
    do
    ./PPMseq -i $i -d 2
    sleep 1
  done
done


echo -e "Iniciando as coletas de dados - CUDA";

for i in `ls images_in/`
  do
  for m in {1,0}
    do  
    for j in $(seq 10);
      do
      ./PPMcuda -i $i -m $m -d 2
      sleep 1
    done
  done
done

