#/bin/bash

NGPU=1
NTEST=5

_CSV_SUMMARY="summary_results.csv"
[ -f ${_CSV_SUMMARY} ] && rm ${_CSV_SUMMARY}
for size in 20000 40000 80000 100000
do
    for kernel in 1 2 3
    do
	_CSV_OUTPUT=generated_matrix_${size}_v${kernel}.csv
	[ -f ${_CSV_OUTPUT} ] && rm ${_CSV_OUTPUT}
        ./test_spmspv g $size $NGPU $NTEST $kernel generated_matrix
        cat ${_CSV_OUTPUT} >> ${_CSV_SUMMARY}
	echo "" >> ${_CSV_SUMMARY}
    done
done
