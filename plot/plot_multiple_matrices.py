#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import sys


def calc_speedup(time_array):
  speedups = np.array([])
  for i in range(time_array.size):
    speedups = np.append(speedups, time_array[0]/time_array[i])
  return speedups


def main(argv):
  matrix_list = ['']
  ngpu = int(argv[0])
  print("input matrix: " + matrix_name)


  part_CSR_baseline = np.array([])
  part_pCSR = np.array([])
  part_pCSR_opt = np.array([])


  part_CSC_baseline = np.array([])
  part_pCSC = np.array([])
  part_pCSC_opt = np.array([])

  part_COO_baseline = np.array([])
  part_pCOO = np.array([])
  part_pCOO_opt = np.array([])


  comp_CSR_baseline = np.array([])
  comp_pCSR = np.array([])

  comp_CSC_baseline = np.array([])
  comp_pCSC = np.array([])

  comp_COO_baseline = np.array([])
  comp_pCOO = np.array([])

  comm_CSR_baseline = np.array([])
  comm_pCSR = np.array([])

  comm_CSC_baseline = np.array([])
  comm_pCSC = np.array([])

  comm_COO_baseline = np.array([])
  comm_pCOO = np.array([])


  merg_CSR_baseline = np.array([])
  merg_pCSR = np.array([])
  merg_pCSR_opt = np.array([])


  merg_CSC_baseline = np.array([])
  merg_pCSC = np.array([])
  merg_pCSC_opt = np.array([])

  merg_COO_baseline = np.array([])
  merg_pCOO = np.array([])
  merg_pCOO_opt = np.array([])


  total_CSR_baseline = np.array([])
  total_pCSR = np.array([])
  total_pCSR_opt = np.array([])


  total_CSC_baseline = np.array([])
  total_pCSC = np.array([])
  total_pCSC_opt = np.array([])

  total_COO_baseline = np.array([])
  total_pCOO = np.array([])
  total_pCOO_opt = np.array([])

  for ngpu in range(1,ngpu+1):
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

    part_CSR_baseline = np.append(part_CSR_baseline, df0.at[0, 'Partition'])
    part_pCSR = np.append(part_pCSR, df0.at[1, 'Partition'])
    part_pCSR_opt = np.append(part_pCSR_opt, df1.at[1, 'Partition'])

    part_CSC_baseline = np.append(part_CSC_baseline, df0.at[3, 'Partition'])
    part_pCSC = np.append(part_pCSC, df0.at[4, 'Partition'])
    part_pCSC_opt = np.append(part_pCSC_opt, df1.at[4, 'Partition'])

    part_COO_baseline = np.append(part_COO_baseline, df0.at[6, 'Partition'])
    part_pCOO = np.append(part_pCOO, df0.at[7, 'Partition'])
    part_pCOO_opt = np.append(part_pCOO_opt, df1.at[7, 'Partition'])


    comp_CSR_baseline = np.append(comp_CSR_baseline, df0.at[0, 'Computation'])
    comp_pCSR = np.append(comp_pCSR, df0.at[1, 'Computation'])

    comp_CSC_baseline = np.append(comp_CSC_baseline, df0.at[3, 'Computation'])
    comp_pCSC = np.append(comp_pCSC, df0.at[4, 'Computation'])

    comp_COO_baseline = np.append(comp_COO_baseline, df0.at[6, 'Computation'])
    comp_pCOO = np.append(comp_pCOO, df0.at[7, 'Computation'])

    comm_CSR_baseline = np.append(comm_CSR_baseline, df0.at[0, 'H2D'])
    comm_pCSR = np.append(comm_pCSR, df0.at[1, 'H2D'])

    comm_CSC_baseline = np.append(comm_CSC_baseline, df0.at[3, 'H2D'])
    comm_pCSC = np.append(comm_pCSC, df0.at[4, 'H2D'])

    comm_COO_baseline = np.append(comm_COO_baseline, df0.at[6, 'H2D'])
    comm_pCOO = np.append(comm_pCOO, df0.at[7, 'Computation'])


    merg_CSR_baseline = np.append(merg_CSR_baseline, df0.at[0, 'Result Merging'])
    merg_pCSR = np.append(merg_pCSR, df0.at[1, 'Result Merging'])
    merg_pCSR_opt = np.append(merg_pCSR_opt, df1.at[1, 'Result Merging'])

    merg_CSC_baseline = np.append(merg_CSC_baseline, df0.at[3, 'Result Merging'])
    merg_pCSC = np.append(merg_pCSC, df0.at[4, 'Result Merging'])
    merg_pCSC_opt = np.append(merg_pCSC_opt, df1.at[4, 'Result Merging'])

    merg_COO_baseline = np.append(merg_COO_baseline, df0.at[6, 'Result Merging'])
    merg_pCOO = np.append(merg_pCOO, df0.at[7, 'Result Merging'])
    merg_pCOO_opt = np.append(merg_pCOO_opt, df1.at[7, 'Result Merging'])

  total_CSR_baseline = part_CSR_baseline + comp_CSR_baseline + comm_CSR_baseline# + merg_CSR_baseline
  total_pCSR = part_pCSR + comp_pCSR + comm_pCSR# + merg_pCSR
  total_pCSR_opt = part_pCSR_opt + comp_pCSR + comm_pCSR# + merg_pCSR_opt

  total_CSC_baseline = part_CSC_baseline + comp_CSC_baseline + comm_CSC_baseline# + merg_CSC_baseline
  total_pCSC = part_pCSC + comp_pCSC + comm_pCSC# + merg_pCSC;
  total_pCSC_opt = part_pCSC_opt + comp_pCSC + comm_pCSC# + merg_pCSC_opt

  total_COO_baseline = part_COO_baseline + comp_COO_baseline + comm_COO_baseline# + merg_COO_baseline
  total_pCOO = part_pCOO + comp_pCOO + comm_pCOO# + merg_pCOO
  total_pCOO_opt = part_pCOO_opt + comp_pCOO + comm_pCOO# + merg_pCOO_opt

  speedup_CSR_baseline = calc_speedup(total_CSR_baseline)
  speedup_pCSR = calc_speedup(total_pCSR)
  speedup_pCSR_opt = calc_speedup(total_pCSR_opt)


  speedup_CSC_baseline = calc_speedup(total_CSC_baseline)
  speedup_pCSC = calc_speedup(total_pCSC)
  speedup_pCSC_opt = calc_speedup(total_pCSC_opt)

  speedup_COO_baseline = calc_speedup(total_COO_baseline)
  speedup_pCOO = calc_speedup(total_pCOO)
  speedup_pCOO_opt = calc_speedup(total_pCOO_opt)



if __name__ == "__main__":
   main(sys.argv[1:])
