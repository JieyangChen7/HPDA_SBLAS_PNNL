#/bin/bash

NGPU=1
NTEST=5

_CSV_SUMMARY="summary_results_generated_matrix.csv"
[ -f ${_CSV_SUMMARY} ] && rm ${_CSV_SUMMARY}
for size in 10000 20000 40000
do
  _CSV_OUTPUT=generated_matrix_${size}.csv
  [ -f ${_CSV_OUTPUT} ] && rm ${_CSV_OUTPUT}
  ./test_spmv g $size $NGPU $NTEST generated_matrix
  cat ${_CSV_OUTPUT} >> ${_CSV_SUMMARY}
  echo "" >> ${_CSV_SUMMARY}
done
