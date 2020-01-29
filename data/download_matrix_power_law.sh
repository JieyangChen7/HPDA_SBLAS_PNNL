#!/bin/bash

download() {
  GROUP=$1
  MATRIX=$2
  wget https://sparse.tamu.edu/MM/${GROUP}/${MATRIX}.tar.gz
  tar -zxf ${MATRIX}.tar.gz
  mv ./${MATRIX}/${MATRIX}.mtx .
  rm -rf ./${MATRIX}/
  rm ${MATRIX}.tar.gz
}

# download Kamvar Stanford
# download Pajek IMDB
# download SNAP wiki-Talk
# download SNAP web-Google
# download Buss connectus
# download Barabasi NotreDame_actors
# download DIMACS10 citationCiteseer
# download SNAP soc-sign-epinions
# download Belcastro human_gene2
# download Belcastro mouse_gene

# download LAW hollywood-2009
# download SNAP com-Orkut
# download SNAP cit-Patents
# download SNAP com-LiveJournal
# download Gleich wb-edu

#download DIMACS10 europe_osm
#download LAW uk-2005

download Fluorem HV15R

