# -----------------------------------------------------------------------------------------
# DESCRIPTION:
#
#    SPMD refers to "Single Program/Multiple Data" parallelization.
# 
#    In CoLM, processes do three types of tasks,
#    1. master : There is only one master process, usually rank 0 in global communicator. 
#                It reads or writes global data, prints informations.
#    2. io     : IO processes read data from files and scatter to workers, gather data from 
#                workers and write to files.
#    3. worker : Worker processes do model calculations.
#   
#    Notice that,
#    1. There are mainly two types of data in CoLM: gridded data and vector data. 
#       Gridded data takes longitude and latitude   as its last two dimensions. 
#       Vector  data takes ELEMENT/PATCH/HRU/PFT/PC as its last dimension.
#       Usually gridded data is allocated on IO processes and vector data is allocated on
#       worker processes.
#    2. One IO process and multiple worker processes form a group. The Input/Output 
#       in CoLM is mainly between IO and workers in the same group. However, all processes
#       can communicate with each other.
#    3. Number of IO is less or equal than the number of blocks with non-zero elements.
#
# Created by Shupeng Zhang, May 2023
# -----------------------------------------------------------------------------------------
from mpi4py import MPI
# import math
import numpy as np


# 并行计算环境初始化
class CoLM_SPMD_Task(object):
    def __init__(self, USEMPI=False, MyComm_r=None) -> None:
        self.MPI_UNDEFINED = -32766
        self.p_root = 0
        self.USEMPI = USEMPI

        self.p_is_master = True
        self.p_is_io = True  # 在divide_processes_into_groups方法有修改
        self.p_is_worker = True

        self.p_np_glb = 1
        self.p_np_worker = 1
        self.p_np_io = 1  # 在divide_processes_into_groups方法有修改

        self.p_iam_glb = 0  # 在divide_processes_into_groups方法有修改
        self.p_iam_io = 0
        self.p_iam_worker = 0

        self.p_np_group = 1
        self.p_iam_glb = 1
        self.p_np_glb = 0
        self.p_iam_group = 0

        # MPI.Init()

        # if MPI.Is_initialized():
        #     MPI.Init()

        # MPI 可用mpi4py代替
        self.p_comm_glb = (MyComm_r if MyComm_r is not None else MPI.COMM_WORLD)

        # 1. Constructing global communicator.
        self.p_iam_glb = self.p_comm_glb.Get_rank()
        self.p_np_glb = self.p_comm_glb.Get_size()

        # print(self.p_iam_glb,self.p_np_glb,'------------')
        self.p_is_master = (self.p_iam_glb == self.p_root)

        self.p_is_writeback = False

    def spmd_assign_writeback(self):

        self.p_comm_glb_plus = self.p_comm_glb

        self.p_comm_glb.Free()

        p_iam_glb_plus = self.p_comm_glb_plus.Get_rank()
        self.p_is_writeback = (p_iam_glb_plus == 0)

        if not self.p_is_writeback:

            # Reconstruct global communicator.
            self.p_comm_glb = self.p_comm_glb_plus.Split(0, p_iam_glb_plus)
            self.p_iam_glb = self.p_comm_glb.Get_rank()
            self.p_np_glb = self.p_comm_glb.Get_size()
            self.p_is_master = (self.p_iam_glb == self.p_root)

        else:
            self.p_comm_glb = self.p_comm_glb_plus.mpi_comm_split(self.MPI_UNDEFINED, p_iam_glb_plus)
            self.p_is_master = False

    def divide_processes_into_groups(self, numblocks, groupsize):
        # 1. Determine number of groups
        ngrp = max((self.p_np_glb - 1) / groupsize, 1)
        ngrp = min(ngrp, numblocks)

        if ngrp <= 0:
            self.p_comm_glb.Barrier()

        # 2. What task will I take? Which group I am in?
        nave = (self.p_np_glb - 1) / ngrp
        nres = (self.p_np_glb - 1) % ngrp

        if not self.p_is_master:
            if self.p_iam_glb <= (nave + 1) * nres:
                self.p_is_io = (self.p_iam_glb % nave + 1) == 1
                self.p_my_group = (self.p_iam_glb - 1) / (nave + 1)
            else:
                self.p_is_io = (self.p_iam_glb - (nave + 1) * nres % nave) == 1
                self.p_my_group = (self.p_iam_glb - (nave + 1) * nres - 1) / nave + nres

            self.p_is_worker = not self.p_is_io
        else:
            self.p_is_io = False
            self.p_is_worker = False
            self.p_my_group = -1

        # 3. Construct IO communicator and address book.
        if self.p_is_io:
            key = 1
            p_comm_io = self.p_comm_glb.Split(key, self.p_iam_glb, )
            self.p_iam_io = p_comm_io.Get_rank()
        else:
            self.p_comm_io = self.p_comm_glb.Split(self.MPI_UNDEFINED, self.p_iam_glb)

        if not self.p_is_io:
            self.p_iam_io = -1
        self.p_itis_io = np.zeros((0, self.p_np_glb - 1))
        self.p_comm_glb.mpi_allgather(self.p_iam_io, self.p_itis_io)

        p_np_io = len(np.where(self.p_itis_io >= 0)[0])
        self.p_address_io = np.zeros((p_np_io - 1))

        for iproc in range(self.p_np_glb - 1):
            if self.p_itis_io[iproc] >= 0:
                self.p_address_io[self.p_itis_io[iproc]] = iproc

        # 4. Construct worker communicator and address book.
        if self.p_is_worker:
            key = 1
            p_comm_worker = self.p_comm_glb.Split(key, self.p_iam_glb)
            self.p_iam_worker = p_comm_worker.Get_rank()
        else:
            p_comm_worker = self.p_comm_glb.Split(self.MPI_UNDEFINED, self.p_iam_glb)

        if not self.p_is_worker:
            self.p_iam_worker = -1
        self.p_itis_worker = np.zeros((self.p_np_glb - 1))
        self.p_comm_glb.mpi_allgather(self.p_iam_worker, self.p_itis_worker)

        p_np_worker = len(np.where(self.p_itis_worker >= 0)[0])

        self.p_address_worker = np.zeros((p_np_worker - 1))

        for iproc in range(self.p_np_glb - 1):
            if self.p_itis_worker[iproc] >= 0:
                self.p_address_worker[self.p_itis_worker[iproc]] = iproc

        # 5. Construct group communicator.
        p_comm_group = self.p_comm_glb.Split(self.p_my_group, self.p_iam_glb)
        self.p_iam_group = p_comm_group.Get_rank()
        self.p_np_group = p_comm_group.Get_size()

        # 6. Print global task informations.
        p_igroup_all = np.zeros((self.p_np_glb - 1))
        self.p_comm_glb.mpi_allgather(self.p_my_group, p_igroup_all)

        if self.p_is_master:

            print('MPI information:')
            print(' Master is ' + str(self.p_root))

            for igrp in range(p_np_io - 1):
                cnum = igrp
                info = 'Group ' + str(cnum) + ' includes '

                cnum = self.p_address_io[igrp]
                info = info + ' IO(' + str(cnum) + '), worker('

                for iproc in range(self.p_np_glb - 1):
                    if (p_igroup_all[iproc] == igrp) and (iproc != self.p_address_io[igrp]):
                        cnum = iproc
                        info = info + cnum

                info = info + ')'
                print(info)
        del p_igroup_all

    def MPI_stop(self, mesg):
        if mesg is not None:
            print(mesg)
        if self.USEMPI:
            self.p_comm_glb.Barrier()
        else:
            exit(1)

    def spmd_exit(self):
        self.release(self.p_itis_io)
        self.release(self.p_address_io)
        self.release(self.p_itis_worker)
        self.release(self.p_address_worker)

        if not self.p_is_writeback:
            self.p_comm_glb.Barrier()

        MPI.Finalize()

    def release(self, o):
        if o is not None:
            del o
