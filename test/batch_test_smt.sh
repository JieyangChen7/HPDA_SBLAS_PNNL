#/bin/bash


for matrix in europe_osm #Stanford IMDB wiki-Talk web-Google connectus NotreDame_actors citationCiteseer soc-sign-epinions human_gene2 mouse_gene hollywood-2009 com-Orkut cit-Patents com-LiveJournal wb-edu
do
  ./submit_test_smt.sh ${matrix}.mtx 
done
