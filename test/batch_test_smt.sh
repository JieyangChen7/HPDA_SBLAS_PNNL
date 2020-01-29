#/bin/bash


for matrix in com-Orkut HV15R #wb-edu mouse_gene hollywood-2009 com-LiveJournal cit-Patents
do
  ./submit_test_smt.sh ${matrix}.mtx 
done
