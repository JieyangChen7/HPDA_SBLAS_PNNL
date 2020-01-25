#! /bin/bash

job_script_template=matrix_test.sh
#job_script=matrix_test_tmp.sh 

for matrix_file in HV15R.mtx #europe_osm.mtx #road_usa.mtx #taly_osm.mtx #great-britain_osm.mtx delaunay_n22.mtx germany_osm.mtx asia_osm.mtx road_central.mtx delaunay_n23.mtx road_usa.mtx kron_g500-logn20.mtx delaunay_n24.mtx
do
  for NGPU in 1 2 3 4 5 6
  do
    job_script=matrix_test_${matrix_file}_${NGPU}.sh
    cp ${job_script_template} ${job_script}
    sed -i 's/MMM/'"${matrix_file}"'/g' ${job_script}
    sed -i 's/GGG/'"${NGPU}"'/g' ${job_script}  
    bsub ${job_script}
  done
done
	 	
