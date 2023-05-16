import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import h5py
import numpy as np
import torch
from models import PartialDeepSeaModel
import train


def get_args():
    parser = argparse.ArgumentParser('Perform test')
    parser.add_argument('model', help='The model')
    parser.add_argument('data_file', help='The data file')
    parser.add_argument('out_pre', help='The output file prefix')
    parser.add_argument('out_dir', help='The output directory')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    model = PartialDeepSeaModel(4, use_weightsum=True, leaky=True, use_sigmoid=False)
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    print('Generating:',args.out_pre)
    (train_data, train_left_data, train_right_data,
     train_left_edges, train_right_edges,
     train_labels, train_dists, train_kmers) = train.load_hdf5_data(args.data_file)

    data_store = h5py.File(os.path.join(args.out_dir, args.out_pre + 'factor_outputs.hdf5'), 'w')
    left_data_store = data_store.create_dataset('left_out', (len(train_labels), model.num_filters[-1] * 2),
                                                dtype='float32',chunks=True, compression='gzip')
    right_data_store = data_store.create_dataset('right_out', (len(train_labels), model.num_filters[-1] * 2),
                                                 dtype='float32',chunks=True, compression='gzip')
    dist_data_store = data_store.create_dataset('dists', data=train_dists, dtype='float32',
                                                chunks=True, compression='gzip')
    pairs = train_data['pairs'][:]
    pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
    pair_data_store = data_store.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,compression='gzip')
    kmers_data_store = data_store.create_dataset('kmers', data=np.array(train_kmers, dtype='float32'), chunks=True, compression='gzip')

    labels_data_store = data_store.create_dataset('labels', data=train_labels, dtype='uint8')
    i = 0
    last_print = 0
    with torch.no_grad():
        while i < len(train_left_edges) - 1:
            #true参数表示用模型进行评估，而不是训练
            end, left_out, right_out, _, _, _ = train.compute_factor_output(train_left_data, train_left_edges,
                                                                                      train_right_data,
                                                                                      train_right_edges,
                                                                                      train_dists, train_labels, train_kmers, i,
                                                                                      True, factor_model=model, max_size=2000, same=False)
            left_data_store[i:end] = left_out.data.cpu().numpy()
            right_data_store[i:end] = right_out.data.cpu().numpy()
            if end - last_print > 5000:
                last_print = end
                print('generating input : %d / %d' % (end, len(train_labels)))
            i = end
    data_store.close()

