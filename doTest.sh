#! /bin/bash

echo -e "Iniciando as coletas de dados";

for img in `ls images_in/`
  do
  for c in {1,0}
  do
    for t in {1,5,8}
      do
      for i in {6,11,16,21}
        do
        for j in $(seq 10);
        do
          mpirun --mca btl_tcp_if_exclude "virbr0" -n $i -hostfile nodes2 ./PPMparapelo -i $img -t $t -a $c -d 2
          sleep 1
        done
      done
    done
  done
done

for img in `ls images_in/`
  do
  for c in {1,0}
  do
    for t in {1,5,8}
      do
      for i in {6,11,16,21,31,41}
        do
        for j in $(seq 10);
        do
          mpirun --mca btl_tcp_if_exclude "virbr0" -n $i -hostfile nodes ./PPMparapelo -i $img -t $t -c $c -d 2
          sleep 1
        done
      done
    done
  done
done

