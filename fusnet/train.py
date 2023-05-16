import h5py
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time
from six.moves import cPickle
import logging
import sys
from collections import Counter
from models import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#功能是，取出当前anchor 对的左右anchor的一些数据
def get_data_batch(left_data, left_edges, right_data, right_edges, dists, labels, kmers, start,
                   max_size=3000, limit_to_one=False):
    #用于记录当前数据结束的edge
    end_idx = start + 1
    # 当前数据开始的edge
    i = start + 1
    #true
    if not limit_to_one:
        #遍历完当前数据的edge
        while i < len(left_edges) - 1:
            #如果左右两个anchor的分区数有一个大于300，则break
            if max(left_edges[i] - left_edges[start], right_edges[i] - right_edges[start]) > max_size:
                break
            i += 1
    #取i-1和end_idx中较大者为end_idx
    end_idx = max(end_idx, i - 1)
    #取出左anchor的序列
    curr_left = left_data[left_edges[start]:left_edges[end_idx]]
    #取出左anchor的分区
    curr_left_edges = left_edges[start:(end_idx + 1)]
    # 取出右anchor的序列
    curr_right = right_data[right_edges[start]:right_edges[end_idx]]
    # 取出右anchor的分区
    curr_right_edges = right_edges[start:(end_idx + 1)]

    #取出当前label和dist
    curr_labels = labels[start:end_idx]
    curr_dists = dists[start:end_idx]
    curr_kmers = kmers[start:end_idx]
    return end_idx, curr_left, curr_left_edges, curr_right, curr_right_edges, curr_dists, curr_labels, curr_kmers


def __get_max(curr_out, start, end):
    curr_max = torch.max(curr_out[start:end, :], dim=0)
    curr_idxes = curr_max[1].data.cpu().numpy()

    counts = Counter(curr_idxes)
    items = sorted(counts.items())
    largest_idx = 0
    max_count = 0
    if len(items) > 4:
        for i in range(len(items) - 4):
            curr_count = sum([x[1] for x in items[i:i+5]])
            if curr_count > max_count:
                max_count = curr_count
                largest_idx = i
    selected_idx = start + largest_idx
    return selected_idx


def compute_one_side(data, edges, model, evaluation=False, same=False):
    x = torch.autograd.Variable(torch.from_numpy(data).float()).cuda()
    #print('x:',x)
    x_rc = torch.autograd.Variable(torch.from_numpy(np.array(data[:,::-1,::-1])).float()).cuda()
    edges = np.array(edges) - edges[0]
    combined = []
    #same=Flase
    if not same:
        curr_outputs = torch.cat((model(x), model(x_rc)), dim=1)
        for i in range(len(edges) - 1):
            combined.append(torch.max(curr_outputs[edges[i]:edges[i + 1], :], dim=0, keepdim=True)[0])
    else:
        f_out = model(x)
        rc_out = model(x_rc)
        for i in range(len(edges)-1):
            selected_idx_f = __get_max(f_out, edges[i], edges[i+1])
            selected_idx_rc = __get_max(rc_out, edges[i], edges[i+1])
            curr_outputs = torch.cat((torch.max(f_out[selected_idx_f:min(selected_idx_f+5, edges[i+1]), :],
                                                dim=0, keepdim=True)[0],
                                      torch.max(rc_out[selected_idx_rc:min(selected_idx_rc+5, edges[i+1]), :],
                                                dim=0, keepdim=True)[0]),
                                     dim=1)
            combined.append(curr_outputs)
    out = torch.cat([x for x in combined], dim=0)
    return out


def compute_factor_output(left_data, left_edges, right_data, right_edges,
                          dists, labels, kmers, start, evaluation, factor_model, max_size=300,
                          limit_to_one=False, same=False, legacy=False):
    (end, curr_left_data, curr_left_edges, curr_right_data,
     curr_right_edges, curr_dists, curr_labels, curr_kmers) = get_data_batch(left_data,
                                                                 left_edges,
                                                                 right_data,
                                                                 right_edges,
                                                                 dists,
                                                                 labels,
                                                                 kmers,
                                                                 start,
                                                                 max_size=max_size,
                                                                 limit_to_one=limit_to_one)
    left_out = compute_one_side(curr_left_data, curr_left_edges, factor_model, evaluation, same=same)
    right_out = compute_one_side(curr_right_data, curr_right_edges, factor_model, evaluation, same=same)
    if legacy:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).long()).cuda()
    else:
        curr_labels = torch.autograd.Variable(torch.from_numpy(curr_labels).float()).cuda()
    curr_dists = torch.autograd.Variable(torch.from_numpy(np.array(curr_dists, dtype='float32'))).cuda()
    curr_kmers = torch.autograd.Variable(torch.from_numpy(np.array(curr_kmers, dtype='float32'))).cuda()
    return end, left_out, right_out, curr_dists, curr_labels, curr_kmers


def apply_classifier(classifier, left_out, right_out, curr_dists, curr_kmers):

    if len(curr_dists.size()) == 1:
        curr_dists = curr_dists.view(-1, 1)
    combined1 = torch.cat((left_out, right_out, curr_dists, curr_kmers), dim=1)

    probs = classifier(combined1)
    return probs


def predict(model, classifier, loss_fn,
            valid_left_data, valid_left_edges,
            valid_right_data, valid_right_edges,
            valid_dists, valid_labels, valid_kmers, return_prob=False, use_distance=True,
            use_metrics=True, max_size=300, verbose=0, same=False, legacy=False):

    model.eval()
    classifier.eval()
    val_err = 0.
    val_samples = 0
    all_probs = []
    last_print = 0
    edge = 0
    with torch.no_grad():
        while edge < len(valid_left_edges) - 1:
            end, left_out, right_out, curr_dists, curr_labels, curr_kmers = compute_factor_output(valid_left_data, valid_left_edges,
                                                                                      valid_right_data, valid_right_edges,
                                                                                      valid_dists, valid_labels, valid_kmers, edge, True,
                                                                                      model, max_size=max_size, same=same, legacy=legacy)
            if verbose > 0:
                logging.info(str(curr_dists.size()))

            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, curr_kmers)
            loss = loss_fn(curr_outputs, curr_labels)
            if legacy:
                if int(torch.__version__.split('.')[1]) > 2:
                    val_predictions = F.softmax(curr_outputs, dim=1).data.cpu().numpy()
                else:
                    val_predictions = F.softmax(curr_outputs).data.cpu().numpy()
                val_predictions = val_predictions[:,1]
            else:
                val_predictions = torch.sigmoid(curr_outputs).data.cpu().numpy()
            all_probs.append(val_predictions)
            val_err += loss.data.item() * (end - edge)

            val_samples += end - edge
            if verbose > 0 and end - last_print > 10000:
                logging.info(str(end))
            edge = end
    #print('all_probs:',all_probs)
    all_probs = np.concatenate(all_probs)

    if use_metrics:
        c_auprc = [metrics.average_precision_score(valid_labels, all_probs), ]
        c_roc = [metrics.roc_auc_score(valid_labels, all_probs), ]

        logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
        logging.info("  auPRCs: {}".format("\t".join(map(str, c_auprc))))
        logging.info("  auROC: {}".format("\t".join(map(str, c_roc))))
        all_preds = np.zeros(all_probs.shape[0])
        all_preds[all_probs > 0.5] = 1
        logging.info("  f1: {}".format(str(metrics.f1_score(valid_labels, all_preds))))
        logging.info("  precision: {}".format(str(metrics.precision_score(valid_labels, all_preds))))
        logging.info("  recall: {}".format(str(metrics.recall_score(valid_labels, all_preds))))
        logging.info("  accuracy: {}".format(str(metrics.accuracy_score(valid_labels, all_preds))))
        logging.info("  ratio: {}".format(np.sum(valid_labels) / len(valid_labels)))
        one_prec = metrics.precision_score(valid_labels, np.ones(len(valid_labels)))
        precision, recall, _ = metrics.precision_recall_curve(valid_labels, all_probs, pos_label=1)
        fpr, tpr, _ = metrics.roc_curve(valid_labels, all_probs, pos_label=1)
    if return_prob:
        return val_err / val_samples, all_probs
    return val_err / val_samples

def load_hdf5_data(fn):
    data = h5py.File(fn, 'r')
    left_data = data['left_data']
    right_data = data['right_data']
    left_edges = data['left_edges'][:]
    right_edges = data['right_edges'][:]
    labels = data['labels'][:]
    pairs = data['pairs'][:]
    kmers = data['kmers'][:]
    dists = [[np.log10(abs(p[5] / 5000 - p[2] / 5000 + p[4] / 5000 - p[1] / 5000) * 0.5) / np.log10(2000001 / 5000),] + list(p)[7:]
                   for p in pairs]
    #print('dists:',dists)
    return data,left_data,right_data,left_edges,right_edges,labels,dists,kmers


def train(model, classifier, data_pre, model_name, retraining, use_existing=None, epochs=60,
          use_weight_for_training=None, init_lr=0.0002, finetune=False, generate_data=True,
          interval=5000, verbose=0, ranges_to_skip=None, use_distance=True, eps=1e-8, same=False,
          model_dir='/data/protein', legacy=False, plot=False):

    model.cuda()
    classifier.cuda()

    #加载训练集和验证集
    #train_data:所有数据
    #train_left_data：anchor1的序列
    #train_left_edges：anchor1/1000的分区数
    print('***********************************Loading data***********************************')
    (train_data, train_left_data, train_right_data,
     train_left_edges, train_right_edges,
     train_labels, train_dists, train_kmers) = load_hdf5_data("%s_train.hdf5" % data_pre)
    (val_data, valid_left_data, valid_right_data,
     valid_left_edges, valid_right_edges,
     valid_labels, valid_dists, valid_kmers) = load_hdf5_data("%s_valid.hdf5" % data_pre)

    # 创建一个logger日志对象
    rootLogger = logging.getLogger()
    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)
    rootLogger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler('logs/' + model_name + "%s.log"%('_re' if retraining else ''))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    logging.info('learning rate: %f, eps: %f' % (init_lr, eps))

    weights = torch.FloatTensor([1, 1]).cuda()
    logging.info(str(weights))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(model.parameters()),
                                lr=init_lr, eps=eps,
                                weight_decay=init_lr * 0.1)

    best_val_loss = predict(model, classifier, loss_fn,
                            valid_left_data, valid_left_edges,
                            valid_right_data, valid_right_edges,
                            valid_dists, valid_labels, valid_kmers, return_prob=False,
                            use_distance=use_distance, verbose=verbose, same=same,
                            legacy=legacy)

    print(best_val_loss)

    last_update = 0
    if ranges_to_skip is not None:
        ranges_to_skip.sort(key=lambda k:(k[0],k[1]))

    for epoch in range(0, epochs):
        start_time = time.time()
        i = 0
        train_loss = 0.
        num_samples = 0

        model.train()
        classifier.train()

        last_print = 0
        curr_loss = 0.
        curr_pos = 0
        print('***********************************Training***********************************')
        while i < len(train_labels):
            end, left_out, right_out, curr_dists, \
            curr_labels, curr_kmers = compute_factor_output(train_left_data,train_left_edges,
                                                              train_right_data,train_right_edges,
                                                              train_dists, train_labels,
                                                              train_kmers, i,False, model,
                                                              same=same, legacy=legacy)
            if ranges_to_skip is not None:
                skip_batch = False
                for (r_s, r_e) in ranges_to_skip:
                    if min(r_e, end) > max(i, r_s):
                        skip_batch = True
                        break
                if skip_batch:
                    i = end
                    continue

            if verbose > 0:
                logging.info(str(curr_dists.size()))
            curr_outputs = apply_classifier(classifier, left_out, right_out, curr_dists, curr_kmers)

            loss = loss_fn(curr_outputs, curr_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_samples += end - i
            curr_loss += loss.data.item() * (end - i)
            train_loss += loss.data.item() * (end - i)
            curr_pos += torch.sum(curr_labels).data.item()
            i = end
            if num_samples < 1000 or num_samples - last_print > interval:
                logging.info("%d  %f  %f  %f  %f", i, time.time() - start_time,
                             train_loss / num_samples, curr_loss / (num_samples - last_print),
                             curr_pos*1.0 / (num_samples - last_print))
                curr_pos = 0
                curr_loss = 0
                last_print = num_samples

        logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
        logging.info("Train loss: %f", train_loss / num_samples)

        val_err = predict(model, classifier, loss_fn,valid_left_data, valid_left_edges,valid_right_data,
                          valid_right_edges,valid_dists, valid_labels,valid_kmers, return_prob=False,
                          use_distance=use_distance, verbose=verbose,same=same, legacy=legacy)

        if val_err < best_val_loss or epoch == 0:
            best_val_loss = val_err
            last_update = epoch
            logging.info("current best val: %f", best_val_loss)
            torch.save(model.state_dict(),
                       "{}/{}{}.model.pt".format(model_dir, model_name, '_re' if retraining else ''),
                       pickle_protocol=cPickle.HIGHEST_PROTOCOL)
            torch.save(classifier.state_dict(),
                       "{}/{}{}.classifier.pt".format(model_dir, model_name, '_re' if retraining else ''),
                       pickle_protocol=cPickle.HIGHEST_PROTOCOL)
        if epoch - last_update >= 10:
            break
    if not (finetune and not generate_data):
        train_data.close()
        val_data.close()
    fileHandler.close()
    consoleHandler.close()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)


