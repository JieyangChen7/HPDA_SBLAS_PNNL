#! /bin/bash

job_script_template=matrix_test.sh
#job_script=matrix_test_tmp.sh 

for matrix_file in $1
do
  for NGPU in 1 2 3 4 5 6
  do
  	PART_OPT=0
	MERG_OPT=0

    job_script=matrix_test_${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}.sh
    cp ${job_script_template} ${job_script}
    sed -i 's/MMM/'"${matrix_file}"'/g' ${job_script}
    sed -i 's/GGG/'"${NGPU}"'/g' ${job_script}
    sed -i 's/PPP/'"${PART_OPT}"'/g' ${job_script}
    sed -i 's/EEE/'"${MERG_OPT}"'/g' ${job_script}
    bsub ${job_script}

    PART_OPT=1
	MERG_OPT=1

    job_script=matrix_test_${matrix_file}_${NGPU}_${PART_OPT}_${MERG_OPT}.sh
    cp ${job_script_template} ${job_script}
    sed -i 's/MMM/'"${matrix_file}"'/g' ${job_script}
    sed -i 's/GGG/'"${NGPU}"'/g' ${job_script}
    sed -i 's/PPP/'"${PART_OPT}"'/g' ${job_script}
    sed -i 's/EEE/'"${MERG_OPT}"'/g' ${job_script} 
    bsub ${job_script}


  done
done
	 	
