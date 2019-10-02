#! /bin/bash

#DATA_PREFIX=/raid/data/SuiteSparse/test_matrices
DATA_PREFIX=./
NGPU=1
NTEST=10

_CSV_SUMMARY="summary_results.csv"
[ -f ${_CSV_SUMMARY} ] && rm ${_CSV_SUMMARY}
for matrix_file in FullChip.mtx ASIC_680k.mtx #consph.mtx denormal.mtx HV15R.mtx shar_te2-b2.mtx webbase-1M.mtx  
do
   for kernel_version in 1 2 3
   do
       _CSV_OUTPUT=${matrix_file}_v${kernel_version}.csv
       [ -f ${_CSV_OUTPUT} ] && rm ${_CSV_OUTPUT}
       ./test_spmv f $DATA_PREFIX/$matrix_file $NGPU $NTEST $kernel_version f ${matrix_file} 
       cat ${_CSV_OUTPUT} >> ${_CSV_SUMMARY}
       echo "" >> ${_CSV_SUMMARY}
    done
done


for matrix_file in ins2.mtx  #europe_osm.mtx pkustk14.mtx roadNet-CA.mtx twitter7.mtx uk-2005.mtx
do
   for kernel_version in 1 2 3
   do
       _CSV_OUTPUT=${matrix_file}_v${kernel_version}.csv
       [ -f ${_CSV_OUTPUT} ] && rm ${_CSV_OUTPUT}
       ./test_spmv f $DATA_PREFIX/$matrix_file $NGPU $NTEST $kernel_version b ${matrix_file}
       cat ${_CSV_OUTPUT} >> ${_CSV_SUMMARY}
       echo "" >> ${_CSV_SUMMARY}
   done
done
