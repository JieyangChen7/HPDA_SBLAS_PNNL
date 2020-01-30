#!/usr/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

import sys


SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def calc_speedup(time_array, baseline):
  speedups = np.array([])
  for i in range(time_array.size):
    speedups = np.append(speedups, baseline/time_array[i])
  return speedups


def main(argv):
  platform = str(argv[0])
  matrix_file = argv[1]
  matrix_name = matrix_file[0:-4]

  ngpu = 0
  gpu_list = []
  if (platform == 'smt'):
    ngpu = 6
    gpu_list = [1,2,3,4,5,6]
  elif (platform == 'dgx1'):
    ngpu = 8
    gpu_list = [1,2,3,4,5,6,7,8]

  print("input matrix: " + matrix_name)
  print("platform: " + platform)


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
  comm_pCSR_numa = np.array([])

  comm_CSC_baseline = np.array([])
  comm_pCSC = np.array([])
  comm_pCSC_numa = np.array([])

  comm_COO_baseline = np.array([])
  comm_pCOO = np.array([])
  comm_pCOO_numa = np.array([])


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



  total_pCSR_numa = np.array([])
  total_pCSR_opt_numa = np.array([])



  total_pCSC_numa = np.array([])
  total_pCSC_opt_numa = np.array([])


  total_pCOO_numa = np.array([])
  total_pCOO_opt_numa = np.array([])

  



  for ngpu in gpu_list:
    part_opt=0
    merg_opt=0
    numa=0
    header = ["NUMA Comm", "Partition", "H2D", "Computation", "Result Merging"]
    csv_file = "./data/{}/{}/{}_{}_{}_{}_{}.csv".format(platform, matrix_name, matrix_file, ngpu, part_opt, merg_opt, numa)
    df0 = pd.read_csv(csv_file, names=header)

    part_opt=1
    merg_opt=1
    numa=0
    header = ["NUMA Comm", "Partition", "H2D", "Computation", "Result Merging"]
    csv_file = "./data/{}/{}/{}_{}_{}_{}_{}.csv".format(platform, matrix_name, matrix_file, ngpu, part_opt, merg_opt, numa)
    df1 = pd.read_csv(csv_file, names=header)

    part_opt=1
    merg_opt=1
    numa=1
    header = ["NUMA Comm", "Partition", "H2D", "Computation", "Result Merging"]
    csv_file = "./data/{}/{}/{}_{}_{}_{}_{}.csv".format(platform, matrix_name, matrix_file, ngpu, part_opt, merg_opt, numa)
    df2 = pd.read_csv(csv_file, names=header)

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
    comm_pCSR_numa = np.append(comm_pCSR_numa, df2.at[1, 'H2D'])

    comm_CSC_baseline = np.append(comm_CSC_baseline, df0.at[3, 'H2D'])
    comm_pCSC = np.append(comm_pCSC, df0.at[4, 'H2D'])
    comm_pCSC_numa = np.append(comm_pCSC_numa, df2.at[4, 'H2D'])

    comm_COO_baseline = np.append(comm_COO_baseline, df0.at[6, 'H2D'])
    comm_pCOO = np.append(comm_pCOO, df0.at[7, 'Computation'])
    comm_pCOO_numa = np.append(comm_pCOO_numa, df2.at[7, 'Computation'])


    merg_CSR_baseline = np.append(merg_CSR_baseline, df0.at[0, 'Result Merging'])
    merg_pCSR = np.append(merg_pCSR, df0.at[1, 'Result Merging'])
    merg_pCSR_opt = np.append(merg_pCSR_opt, df1.at[1, 'Result Merging'])

    merg_CSC_baseline = np.append(merg_CSC_baseline, df0.at[3, 'Result Merging'])
    merg_pCSC = np.append(merg_pCSC, df0.at[4, 'Result Merging'])
    merg_pCSC_opt = np.append(merg_pCSC_opt, df1.at[4, 'Result Merging'])

    merg_COO_baseline = np.append(merg_COO_baseline, df0.at[6, 'Result Merging'])
    merg_pCOO = np.append(merg_pCOO, df0.at[7, 'Result Merging'])
    merg_pCOO_opt = np.append(merg_pCOO_opt, df1.at[7, 'Result Merging'])

  total_CSR_baseline = part_CSR_baseline + comp_CSR_baseline + comm_CSR_baseline + merg_CSR_baseline
  total_pCSR = part_pCSR + comp_pCSR + comm_pCSR + merg_pCSR
  total_pCSR_opt = part_pCSR_opt + comp_pCSR + comm_pCSR + merg_pCSR_opt

  total_pCSR_opt_numa = part_pCSR_opt + comp_pCSR + comm_pCSR_numa + merg_pCSR_opt

  total_CSC_baseline = part_CSC_baseline + comp_CSC_baseline + comm_CSC_baseline + merg_CSC_baseline
  total_pCSC = part_pCSC + comp_pCSC + comm_pCSC + merg_pCSC;
  total_pCSC_opt = part_pCSC_opt + comp_pCSC + comm_pCSC + merg_pCSC_opt

  total_pCSC_opt_numa = part_pCSC_opt + comp_pCSC + comm_pCSC_numa + merg_pCSC_opt

  total_COO_baseline = part_COO_baseline + comp_COO_baseline + comm_COO_baseline# + merg_COO_baseline
  total_pCOO = part_pCOO + comp_pCOO + comm_pCOO + merg_pCOO
  total_pCOO_opt = part_pCOO_opt + comp_pCOO + comm_pCOO + merg_pCOO_opt

  total_pCOO_opt_numa = part_pCOO_opt + comp_pCOO + comm_pCOO_numa + merg_pCOO_opt

  speedup_CSR_baseline = calc_speedup(total_CSR_baseline, total_CSR_baseline[0])
  speedup_pCSR = calc_speedup(total_pCSR, total_CSR_baseline[0])
  speedup_pCSR_opt = calc_speedup(total_pCSR_opt, total_CSR_baseline[0])

  speedup_pCSR_opt_numa = calc_speedup(total_pCSR_opt_numa, total_CSR_baseline[0])


  speedup_CSC_baseline = calc_speedup(total_CSC_baseline, total_CSR_baseline[0])
  speedup_pCSC = calc_speedup(total_pCSC, total_CSR_baseline[0])
  speedup_pCSC_opt = calc_speedup(total_pCSC_opt, total_CSR_baseline[0])

  speedup_pCSC_opt_numa = calc_speedup(total_pCSC_opt_numa, total_CSR_baseline[0])

  speedup_COO_baseline = calc_speedup(total_COO_baseline, total_CSR_baseline[0])
  speedup_pCOO = calc_speedup(total_pCOO, total_CSR_baseline[0])
  speedup_pCOO_opt = calc_speedup(total_pCOO_opt, total_CSR_baseline[0])

  speedup_pCOO_opt_numa = calc_speedup(total_pCOO_opt_numa, total_CSR_baseline[0])

############# Constant #############

  xticklabels = ''

  if (ngpu == 6):
    xticklabels = gpu_list
  elif (ngpu == 8):
    xticklabels = gpu_list

  # x_idx = []
  # for i in range(ngpu):
  #   x_idx.append(str(i+1))

############# Parition Time ###############
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  x_idx = np.array(gpu_list)
  
  p1 = ax1.bar(x_idx - width, part_CSR_baseline, width)
  p2 = ax1.bar(x_idx, part_pCSR, width)
  p3 = ax1.bar(x_idx + width, part_pCSR_opt, width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Time (s)")
  ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, part_CSC_baseline.tolist(), width)
  p2 = ax2.bar(x_idx, part_pCSC.tolist(), width)
  p3 = ax2.bar(x_idx + width, part_pCSC_opt.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, part_COO_baseline.tolist(), width)
  p2 = ax3.bar(x_idx, part_pCOO.tolist(), width)
  p3 = ax3.bar(x_idx + width, part_pCOO_opt.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax3.set_title("COO")

  ax1.legend((p1[0], p2[0], p3[0]), ('Baseline', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.3, -0.08), ncol=3)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_partition_time_{}.pdf'.format(matrix_name, platform))  

############# Parition Overhead ###############
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, part_CSR_baseline/(part_CSR_baseline + comp_CSR_baseline + comm_CSR_baseline).tolist(), width)
  p2 = ax1.bar(x_idx, part_pCSR/(part_pCSR + comp_pCSR + comm_pCSR).tolist(), width)
  p3 = ax1.bar(x_idx + width, part_pCSR_opt/(part_pCSR_opt + comp_pCSR + comm_pCSR).tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Overhead")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, part_CSC_baseline/(part_CSC_baseline + comp_CSC_baseline + comm_CSC_baseline).tolist(), width)
  p2 = ax2.bar(x_idx, part_pCSC/(part_pCSC + comp_pCSC + comm_pCSC).tolist(), width)
  p3 = ax2.bar(x_idx + width, part_pCSC_opt/(part_pCSC_opt + comp_pCSC + comm_pCSC).tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, part_COO_baseline/(part_COO_baseline + comp_COO_baseline + comm_COO_baseline).tolist(), width)
  p2 = ax3.bar(x_idx, part_pCOO/(part_pCOO + comp_pCOO + comm_pCOO).tolist(), width)
  p3 = ax3.bar(x_idx + width, part_pCOO_opt/(part_pCOO_opt + comp_pCOO + comm_pCOO).tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax3.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (1, 1.01), ncol=1)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_partition_overhead_{}.pdf'.format(matrix_name, platform))


  ############# Merging Time ###############
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, merg_CSR_baseline.tolist(), width)
  p2 = ax1.bar(x_idx, merg_pCSR.tolist(), width)
  p3 = ax1.bar(x_idx + width, merg_pCSR_opt.tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax1.set_ylabel("Time (s)")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, merg_CSC_baseline.tolist(), width)
  p2 = ax2.bar(x_idx, merg_pCSC.tolist(), width)
  p3 = ax2.bar(x_idx + width, merg_pCSC_opt.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, merg_COO_baseline.tolist(), width)
  p2 = ax3.bar(x_idx, merg_pCOO.tolist(), width)
  p3 = ax3.bar(x_idx + width, merg_pCOO_opt.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
  ax3.set_title("COO")

  ax1.legend((p1[0], p2[0], p3[0]), ('Baseline', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.3, -0.08), ncol=3)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_merge_time_{}.pdf'.format(matrix_name, platform))  

############# Merging Overhead ###############
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, merg_CSR_baseline/(merg_CSR_baseline + comp_CSR_baseline + comm_CSR_baseline).tolist(), width)
  p2 = ax1.bar(x_idx, part_pCSR/(merg_pCSR + comp_pCSR + comm_pCSR).tolist(), width)
  p3 = ax1.bar(x_idx + width, merg_pCSR_opt/(merg_pCSR_opt + comp_pCSR + comm_pCSR).tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Overhead")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, merg_CSC_baseline/(merg_CSC_baseline + comp_CSC_baseline + comm_CSC_baseline).tolist(), width)
  p2 = ax2.bar(x_idx, merg_pCSC/(merg_pCSC + comp_pCSC + comm_pCSC).tolist(), width)
  p3 = ax2.bar(x_idx + width, merg_pCSC_opt/(merg_pCSC_opt + comp_pCSC + comm_pCSC).tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, merg_COO_baseline/(merg_COO_baseline + comp_COO_baseline + comm_COO_baseline).tolist(), width)
  p2 = ax3.bar(x_idx, merg_pCOO/(merg_pCOO + comp_pCOO + comm_pCOO).tolist(), width)
  p3 = ax3.bar(x_idx + width, merg_pCOO_opt/(merg_pCOO_opt + comp_pCOO + comm_pCOO).tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax3.legend((p1[0], p2[0], p3[0]), ('Baseline', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (1, 1.01), ncol=1)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_merg_overhead_{}.pdf'.format(matrix_name, platform))


################ Comm Time #################
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, comm_CSR_baseline.tolist(), width)
  p2 = ax1.bar(x_idx, comm_pCSR.tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Time (s)")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, comm_CSC_baseline.tolist(), width)
  p2 = ax2.bar(x_idx, comm_pCSC.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, comm_COO_baseline.tolist(), width)
  p2 = ax3.bar(x_idx, comm_pCOO.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax3.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*'), loc='upper left', bbox_to_anchor= (1, 1.01), ncol=1)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_comm_time_{}.pdf'.format(matrix_name, platform))  

  ################ Overall Time #################
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, total_CSR_baseline.tolist(), width)
  p2 = ax1.bar(x_idx, total_pCSR.tolist(), width)
  p3 = ax1.bar(x_idx + width, total_pCSR_opt.tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Time (s)")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, total_CSC_baseline.tolist(), width)
  p2 = ax2.bar(x_idx, total_pCSC.tolist(), width)
  p3 = ax2.bar(x_idx + width, total_pCSC_opt.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, total_COO_baseline.tolist(), width)
  p2 = ax3.bar(x_idx, total_pCOO.tolist(), width)
  p3 = ax3.bar(x_idx + width, total_pCOO_opt.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax3.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (1, 1.01), ncol=1)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_total_time_{}.pdf'.format(matrix_name, platform))  

  ################ Overall Speedup #################
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width, speedup_CSR_baseline.tolist(), width)
  p2 = ax1.bar(x_idx, speedup_pCSR.tolist(), width)
  p3 = ax1.bar(x_idx + width, speedup_pCSR_opt_numa.tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Speedup")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width, speedup_CSC_baseline.tolist(), width)
  p2 = ax2.bar(x_idx, speedup_pCSC.tolist(), width)
  p3 = ax2.bar(x_idx + width, speedup_pCSC_opt_numa.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width, speedup_COO_baseline.tolist(), width)
  p2 = ax3.bar(x_idx, speedup_pCOO.tolist(), width)
  p3 = ax3.bar(x_idx + width, speedup_pCOO_opt_numa.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax1.legend((p1[0], p2[0], p3[0]), ('Baseline', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.3, -0.08), ncol=3)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_total_speedup_{}.pdf'.format(matrix_name, platform))  

  ################ NUMA Speedup #################
  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
  width = 0.25 
  #x_idx = ['1','2','3','4','5','6']
  #x_idx = np.arange(ngpu)
  
  p1 = ax1.bar(x_idx - width/2, speedup_pCSR_opt.tolist(), width)
  p2 = ax1.bar(x_idx + width/2, speedup_pCSR_opt_numa.tolist(), width)
  ax1.set_xticks(x_idx)
  ax1.set_xticklabels(xticklabels)
  ax1.set_ylabel("Speedup")
  ax1.set_title("CSR")
  #ax1.legend((p1[0], p2[0], p3[0]), ('Naive', 'p*', 'p*-opt'), loc='upper left', bbox_to_anchor= (-0.55, 1), ncol=1)
  
  p1 = ax2.bar(x_idx - width/2, speedup_pCSC_opt.tolist(), width)
  p2 = ax2.bar(x_idx + width/2, speedup_pCSC_opt_numa.tolist(), width)
  ax2.set_xticks(x_idx)
  ax2.set_xticklabels(xticklabels)
  ax2.set_xlabel("Number of GPUs")
  ax2.set_title("CSC")
  #ax2.legend((p1[0], p2[0], p3[0]), ('CSC-naive', 'pCSC', 'pCSC-opt'), loc='lower left', bbox_to_anchor= (-0.2, 1.01), ncol=3)
  

  p1 = ax3.bar(x_idx - width/2, speedup_pCOO_opt.tolist(), width)
  p2 = ax3.bar(x_idx + width/2, speedup_pCOO_opt_numa.tolist(), width)
  ax3.set_xticks(x_idx)
  ax3.set_xticklabels(xticklabels)
  ax3.set_title("COO")

  ax1.legend((p1[0], p2[0]), ('w/o NUMA-aware', 'w/ NUMA-aware'), loc='upper left', bbox_to_anchor= (0, -0.1), ncol=1)
  
  plt.tight_layout()
  #plt.show()
  plt.savefig('{}_numa_speedup_{}.pdf'.format(matrix_name, platform))  

if __name__ == "__main__":
   main(sys.argv[1:])
