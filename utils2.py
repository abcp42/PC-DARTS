""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc


import os
import sys
#import tensorflow as tf
import numpy as np
from PIL import Image
from obspy.io.segy.segy import _read_segy
import matplotlib.pyplot as plt
import cv2

class SegyReader(object):

    def __init__(self, path, labels_path, batch_size, file_extension=".sgy", max_size=-1):
        """
        :param path:
        :param labels_path:
        :param batch_size:
        :param file_extension:
        :param max_size:
        """
        self._idx = None
        self.path = path
        self.labels_path = labels_path
        self._paths = list()
        self._labels = list()
        self._file_extension = file_extension
        self._max_size = max_size
        self._batch_size = batch_size
        self.load_data()

    def load_data(self):
        with open(self.labels_path, 'r') as f:
            for row in f:
                name, label = row.split(",")
                label = label.lower().strip()
                name = name.strip().replace(".png", ".segy")
                self._paths.append(os.path.join(self.path, name))
                if label == "good":
                    self._labels.append(0)
                elif label == "bad":
                    self._labels.append(1)
                elif label == "ugly":
                    self._labels.append(2)
                else:
                    raise ValueError("Label not recognized. Found in data: '{}'".format(label))
                if 0 < self._max_size == len(self._paths):
                    break

        self._paths = np.asarray(self._paths)
        self._labels = np.asarray(self._labels)

    def load_from_dir(self):
        self._paths = list()
        self._labels = list()
        for root, _, files in os.walk(self.path):
            for file in files:
                if not file.endswith(self._file_extension):
                    continue
                file_path = os.path.join(root, file)
                self._paths.append(file_path)
                self._labels.append(0)

                if 0 < self._max_size == len(self._paths):
                    break
        self._paths = np.asarray(self._paths)
        self._labels = np.asarray(self._labels)

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._paths[item], self._labels[item]

        elif isinstance(item, slice):
            return self._paths[item], self._labels[item]

    def __iter__(self):
        """
        Iterator initializer.
        """
        self._idx = 0
        return self

    def __next__(self):
        """
        Returns the iterator's next element.
        """
        mod_batch = len(self) % self._batch_size
        if self._idx >= (len(self) - mod_batch):

            perm = np.random.permutation(len(self._paths))
            self._paths = self._paths[perm]
            self._labels = self._labels[perm]

            raise StopIteration()

        x = self.load_img(self._paths[self._idx])
        y = self._labels[self._idx]
        # index sum
        self._idx += 1
        return x, y

    def make_dataset(self):
        """
        Returns a tensorflow Dataset created from this current iterator.
        :return: a `tf.data.Dataset`.
        """
        batch_size = self._batch_size
        prefetch_buffer = 10
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(self),
            output_types=(tf.float32, tf.int32),
            # output_shapes=self._inputs_config["output_shapes"]
        )
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(buffer_size=prefetch_buffer)

    def load_img(self, img_path):
        """
        reads and normalize a seismogram from the given segy file.
        :param img_path: a path to the segy file.
        :return: seismogram image as numpy array normalized between 0-1.
        """
        segy = _read_segy(img_path)
        _traces = list()
        for trace in segy.traces:
            _traces.append(trace.data)
        x = np.asarray(_traces, dtype=np.float32)
        std = x.std()
        x -= x.mean()
        x /= std
        x *= 0.1
        x += .5
        x = np.clip(x, 0, 1)

        return x.T
    

import numpy as np
from sklearn.model_selection import StratifiedKFold

def kfold(files,labels,nfolds = 5, nsplit = 5):
  X = np.asarray(files)
  y = np.asarray(labels)
  skf = StratifiedKFold(n_splits=nfolds, random_state= 33,shuffle = True)
  skf.get_n_splits(X, y)
  i = 1
  for train_index, test_index in skf.split(X, y):
    if(i!=nsplit):
      i+=1
      continue
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    break
  return X_train,X_test,y_train,y_test


def get_data(dataset, data_path,val1_data_path,val2_data_path, cutout_length, validation,validation2 = False,n_class = 3,image_size = 64):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif dataset == 'custom':
        dset_cls = dset.ImageFolder
        n_classes = n_class #2 to mama
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length,image_size)
    if dataset == 'custom':
        print("DATA PATH:", data_path)
        #trn_data = dset_cls(root=data_path, transform=trn_transform)
        reader = SegyReader(
            path=data_path,
            labels_path=data_path+"../labels.txt",
            batch_size=1
        )
        Xs = []
        Ys = []
        for i in range(len(reader)):
            x_path, y = reader[i]
            Xs.append(x_path)
            Ys.append(y)
        
            
           
        X_train,X_test,y_train,y_test = kfold(Xs,Ys,2,1)#dividido em 5 folds, 1 forma de fold
        
        x_train_data = []
        x_test_data = []
        for x_path in X_train:
            x = reader.load_img(x_path)
            x_re = cv2.resize(x,(image_size,image_size))
            rgb = cv2.merge([x_re,x_re,x_re])
            x_train_data.append(rgb)
        for x_path in X_test:
            x = reader.load_img(x_path)
            x_re = cv2.resize(x,(image_size,image_size))
            rgb = cv2.merge([x_re,x_re,x_re])
            x_test_data.append(rgb)
            
        x_train_data = np.asarray(x_train_data)
        x_test_data = np.asarray(x_test_data)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        num_classes = 3
        y_train  = (y_train == torch.arange(num_classes).reshape(1, num_classes)).float()
        y_test = (y_test == torch.arange(num_classes).reshape(1, num_classes)).float()

            
        print(y_train)
        tensor_train_x = torch.stack([torch.Tensor(i) for i in x_train_data]) # transform to torch tensors
        tensor_train_y = torch.stack([torch.Tensor(i) for i in y_train])
        tensor_test_x = torch.stack([torch.Tensor(i) for i in x_test_data]) # transform to torch tensors
        tensor_test_y = torch.stack([torch.Tensor(i) for i in y_test])

        train_dataset = utils.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
        test_dataset = utils.TensorDataset(tensor_test_x,tensor_test_y) # create your datset
        #train_dataloader = utils.DataLoader(train_dataset) # create your dataloader
        #dataset_loader = torch.utils.data.DataLoader(trn_data,
        #                                     batch_size=16, shuffle=True,
        #                                     num_workers=1)
        
    else:
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    if dataset == 'custom':
        shape = [1, image_size, image_size,3]
    else:
        shape = trn_data.train_data.shape
    print(shape)
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]
    print('input_size: uitls',input_size)
    
    ret = [input_size, input_channels, n_classes, train_dataset,test_dataset]
    
    """
    ret = [input_size, input_channels, n_classes, trn_data]
     
    if validation: # append validation data
        if dataset == 'custom':
            dset_cls = dset.ImageFolder(val1_data_path,transform=val_transform)
            ret.append(dset_cls)
        else:
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))
    if validation2:
        if dataset == 'custom':
            dset_cls = dset.ImageFolder(val2_data_path,transform=trn_transform)
            ret.append(dset_cls)
    """
    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    #print('output:',output)
    #print('target:',target)
    #print('maxk:',maxk)
    ###TOP 5 NAO EXISTE NAS MAAMAS OU NO GEO. TEM QUE TRATAR
    maxk = 3 # Ignorando completamente o top5

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, ckpt_dir, is_best=False, is_best_overall =False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'w_optimizer_state_dict': w_optimizer.state_dict(),
            'a_optimizer_state_dict': a_optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
    if is_best_overall:
        best_filename = os.path.join(ckpt_dir, 'best_overall.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, filename='checkpoint.pth.tar'):
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer_state_dict'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,w_optimizer,a_optimizer,loss



def save_checkpoint2(model,epoch,optimizer,loss, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'model.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint2(model,epoch,optimizer,loss, filename='model.pth.tar'):
    filename=filename+'checkpoint.pth.tar'
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,optimizer,loss
