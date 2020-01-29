#/bin/bash

#all=Stanford IMDB wiki-Talk web-Google connectus NotreDame_actors citationCiteseer soc-sign-epinions human_gene2 mouse_gene hollywood-2009 com-Orkut cit-Patents com-LiveJournal wb-edu
#large=cit-Patents com-LiveJournal com-Orkut hollywood-2009 human_gene2 mouse_gene wb-edu
for matrix in  com-Orkut HV15R #mouse_gene #hollywood-2009 com-LiveJournal cit-Patents wb-edu
do
  ./submit_test_dgx1.sh ${matrix}.mtx 
done
