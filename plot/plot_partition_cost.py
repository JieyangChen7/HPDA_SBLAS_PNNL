#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import sys




def main(argv):
  matrix_name = argv[0];
  print("input matrix: " + matrix_name)
  part_CSR_baseline = []
  part_pCSR = []
  part_pCSR_opt = []

  part_CSC_baseline = []
  part_pCSC = []
  part_pCSC_opt = []

  part_COO_baseline = []
  part_pCOO = []
  part_pCOO_opt = []
  for ngpu in range(1,7):
    part_opt=0
    merg_opt=0
    header = ["NUMA Comm", "Partition", "H2D", "Computation", "Result Merging"]
    csv_file = "./data/{}_{}_{}_{}.csv".format(matrix_name, ngpu, part_opt, merg_opt)
    df0 = pd.read_csv(csv_file, names=header)

    part_opt=1
    merg_opt=1
    header = ["NUMA Comm", "Partition", "H2D", "Computation", "Result Merging"]
    csv_file = "./data/{}_{}_{}_{}.csv".format(matrix_name, ngpu, part_opt, merg_opt)
    df1 = pd.read_csv(csv_file, names=header)

    part_CSR_baseline.append(df0.at[0, 'Partition'])
    part_pCSR.append(df0.at[1, 'Partition'])
    part_pCSR_opt.append(df1.at[1, 'Partition'])

    part_CSC_baseline.append(df0.at[3, 'Partition'])
    part_pCSC.append(df0.at[4, 'Partition'])
    part_pCSC_opt.append(df1.at[4, 'Partition'])

    part_COO_baseline.append(df0.at[6, 'Partition'])
    part_pCOO.append(df0.at[7, 'Partition'])
    part_pCOO_opt.append(df1.at[7, 'Partition'])

  fig, ax = plt.subplots()
  width = 0.25 
  x_idx = ['1','2','3','4','5','6']
  x_idx = np.arange(6)
  p1 = ax.bar(x_idx, part_CSR_baseline, width)
  p2 = ax.bar(x_idx + width, part_pCSR, width)
  p3 = ax.bar(x_idx + width*2, part_pCSR_opt, width)
  ax.set_xticks(x_idx + width / 2)
  ax.set_xticklabels(('1', '2', '3', '4', '5', '6'))
  plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
