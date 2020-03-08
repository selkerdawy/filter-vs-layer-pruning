from __future__ import absolute_import
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import logging
import os
import sys
import time
import torch
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig','FileLogger', 'TensorboardLogger']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class TensorboardLogger:
  def __init__(self, output_dir, is_master=False):
    self.output_dir = output_dir
    self.current_step = 0
    if is_master: self.writer = SummaryWriter(self.output_dir)
    else: self.writer = NoOp()
    self.log('first', time.time())

  def log(self, tag, val):
    """Log value to tensorboard (relies on global_example_count being set properly)"""
    if not self.writer: return
    self.writer.add_scalar(tag, val, self.current_step)

  def update_step_count(self, batch_total):
    self.current_step += batch_total

  def close(self):
    self.writer.export_scalars_to_json(self.output_dir+'/scalars.json')
    self.writer.close()

  def flush(self):
    if not self.writer: return
    self.writer.flush()

  # Convenience logging methods
  def log_size(self, bs=None, sz=None):
    if bs: self.log('sizes/batch', bs)
    if sz: self.log('sizes/image', sz)

  def log_acc_loss(self, dic):
    for k,v in dic.items():
        self.log("losses/%s"%k, v)

  def log_grad_hist(self, dic):
    for k,v in dic.items():
        self.writer.add_histogram("mask_grads/%s"%k, v.cpu().detach().type(torch.float32), self.current_step)

  def log_memory(self):
    if not self.writer: return
    self.log("memory/allocated_gb", torch.cuda.memory_allocated()/1e9)
    self.log("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9)
    self.log("memory/cached_gb", torch.cuda.memory_cached()/1e9)
    self.log("memory/max_cached_gb", torch.cuda.max_memory_cached()/1e9)

  def log_trn_times(self, batch_time, data_time, batch_size):
    if not self.writer: return
    self.log("times/step", 1000*batch_time)
    self.log("times/data", 1000*data_time)
    images_per_sec = batch_size/batch_time
    self.log("times/1gpu_images_per_sec", images_per_sec)
    self.log("times/8gpu_images_per_sec", 8*images_per_sec)

#based on https://groups.google.com/forum/#!topic/comp.lang.python/0lqfVgjkc68

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        bufsize = 1
        self.log = open(filename, "w", buffering=bufsize)

    def delink(self):
        self.log.close()
        self.log = open('foo', "w")
#        self.write = self.writeTerminalOnly

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)

class FileLogger:
  def __init__(self, output_dir, is_master=False, is_rank0=False):
    self.output_dir = output_dir
    self.is_master = is_master

  def set_local_rank(self, rank):
    # Log to console if rank 0, Log to console and file if master
    if rank!=0 and not (rank is None): self.logger = NoOp()
    else: self.logger = self.get_logger(self.output_dir, log_to_file=self.is_master)

  def get_logger(self, output_dir, log_to_file=True):
    logger = logging.getLogger('imagenet_training')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    if log_to_file:
      vlog = logging.FileHandler(output_dir+'/verbose.log')
      vlog.setLevel(logging.INFO)
      vlog.setFormatter(formatter)
      logger.addHandler(vlog)

      eventlog = logging.FileHandler(output_dir+'/event.log')
      eventlog.setLevel(logging.WARN)
      eventlog.setFormatter(formatter)
      logger.addHandler(eventlog)

      time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
      debuglog = logging.FileHandler(output_dir+'/debug.log')
      debuglog.setLevel(logging.DEBUG)
      debuglog.setFormatter(time_formatter)
      logger.addHandler(debuglog)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

  def console(self, *args):
    self.logger.debug(*args)

  def event(self, *args):
    self.logger.warn(*args)

  def verbose(self, *args):
    self.logger.info(*args)

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op

if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt',
    'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    }

    field = ['Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')
