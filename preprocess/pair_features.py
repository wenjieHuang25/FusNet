# coding: utf-8

import numpy as np
import pylab as pl
import h5py
from sklearn.utils import shuffle
import bisect
import os
os.environ['CUDA_VISIBLE_DEVICE']='3'
import read_reference


CHANNELS_FIRST = "channels_first"
CHANNELS_LAST = "channels_last"

#将碱基序列转换成四维向量
def get_seq_matrix(seq, seq_len: int, data_format: str, one_d: bool, rc=False):
    channels = 4
    # seq长度*4列
    mat = np.zeros((seq_len, channels), dtype="float32")

    for i, a in enumerate(seq):
        idx = i
        if idx >= seq_len:
            break
        a = a.lower()

        if a == 'a':
            mat[idx, 0] = 1
        elif a == 'g':
            mat[idx, 1] = 1
        elif a == 'c':
            mat[idx, 2] = 1
        elif a == 't':
            mat[idx, 3] = 1
        else:
            mat[idx, 0:4] = 0

    if rc:
#行全部倒序，列也全部倒序
        mat = mat[::-1, ::-1]
    # true
    if not one_d:
        # 压缩成1维向量
        mat = mat.reshape((1, seq_len, channels))
    # true
    if data_format == CHANNELS_FIRST:
        axes_order = [len(mat.shape)-1,] + [i for i in range(len(mat.shape)-1)]
#做一个转置，mat变成4列若干行
        mat = mat.transpose(axes_order)

    return mat


# 如果anchor长度<min_size，则需要扩展长度，最后返回从start到end的碱基seq
def _get_sequence(chrom, start, end, min_size=1000, crispred=None):
    # assumes the CRISPRed regions were not overlapping
    # assumes the CRISPRed regions were sorted
#is None
    if crispred is not None:
        #print('crispred is not None')
        seq = ''
        curr_start = start
        for cc, cs, ce in crispred:
            # overlapping
            #print('check', chrom, start, end, cc, cs, ce)
            if chrom == cc and min(end, ce) > max(cs, curr_start):
                #print('over', curr_start, end, cs, ce)
                if curr_start > cs:
                    seq += read_reference.hg19[chrom][curr_start:cs]
                curr_start = ce
        if curr_start < end:
            seq += read_reference.hg19[chrom][curr_start:end]
        #print(start, end, end-start, len(seq))

    else:
        # 从hg19的对应染色体上，取出anchor从开始到结束的碱基seq
        seq = read_reference.hg19[chrom][start:end]
    # 如果取出的seq比min_size还要短
    if len(seq) < min_size:
        # diff = seq比min_size短多少
        diff = min_size - (end - start)
        # 将diff除于2，左边和右边各扩展ext_left的长度
        ext_left = diff // 2
        # 如果左边扩展以后越界了，则不扩展
        if start - ext_left < 0:
            ext_left = start
        # 如果右边扩展以后越界，则ext_left=diff-右边能扩展的最大长度
        elif diff - ext_left + end > len(read_reference.hg19[chrom]):
            ext_left = diff - (len(read_reference.hg19[chrom]) - end)
        # 经过扩展后的起始位点
        curr_start = start - ext_left
        curr_end = end + diff - ext_left

        # 如果起始位点扩展了，则seq也要跟着扩展
        if curr_start < start:
            seq = read_reference.hg19[chrom][curr_start:start] + seq
        if curr_end > end:
            seq = seq + read_reference.hg19[chrom][end:curr_end]
    if start < 0 or end > len(read_reference.hg19[chrom]):
        return None
    return seq


def encode_seq(chrom, start, end, min_size=1000, crispred=None):
#_get_sequence：如果anchor长度<min_size，则需要扩展长度，最后返回从start到end的碱基seq
    seq = _get_sequence(chrom, start, end, min_size, crispred)
    if seq is None:
        return None
    mat = get_seq_matrix(seq, len(seq), 'channels_first', one_d=True, rc=False)
    parts = []
    for i in range(0, len(seq), 500):
        if i + 1000 >= len(seq):
            break
        parts.append(mat[:, i:i + 1000])
    parts.append(mat[:, -1000:])
    parts = np.array(parts, dtype='float32')
    return parts









