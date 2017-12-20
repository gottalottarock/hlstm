import copy
import random
import datetime
import numpy as np
from operator import itemgetter
from .logger import Logger
from .hlstm_model import HLSTMModel
from .tree_lstm import BinaryTreeLSTM

class HLSTMInterface:

    def __init__(self,sess, model, logger=None):
        if not logger:
            logger = Logger()
        self.sess = sess
        self.model = model
        self.logger = logger

    # def train input:
    #   train_set and test_set: [label, [tree, tree, tree, tree]]

    @classmethod
    def init_from_file_and_restore(cls,sess, path_to_model, vocab, tree_cls=BinaryTreeLSTM,
                           model_cls=HLSTMModel, logger = None):
        interface = cls.init_from_file(sess, path_to_model, vocab, tree_cls,
                                       model_cls, logger)
        interface.restore_model(path_to_model)
        return interface

    @classmethod
    def init_from_file(cls,sess, path_to_model,vocab,tree_cls = BinaryTreeLSTM,
                      model_cls=HLSTMModel, logger = None):
        tree_lstm = tree_cls.init_from_file(path_to_model,vocab)
        model = model_cls.init_from_file(path_to_model,tree_lstm)
        interface = cls(sess,model,logger)
        return interface


    def train(self, train_set, test_set=None, epochs=10, dev_batch_size=1,
              save=True, save_model_dir='', sess_name='', func_set_info_dict=None,
              quiet_saver = False, **property_dict):
        if not self.model.is_compiled:
            return -1
        if not func_set_info_dict:
            func_set_info_dict = self.get_set_info
        if not sess_name:
            sess_name = 'train_' + datetime.datetime.now().strftime('%m_%d_%H_%M')
        self.logger.start_session(sess_name)
        self.model.prepare_training(self.sess,**property_dict)
        train_epochs = self.model.train_epochs(self.sess,train_set=train_set, dev_set=test_set,
                                               dev_batch_size=dev_batch_size,
                                               epochs=epochs, save=save,
                                               save_dir=save_model_dir,
                                               quiet_saver = quiet_saver)
        for epoch_res, all_res in train_epochs:
            pass
        self.logger.record_train_logs(model_properies=self.model.model_properties,
                                      train_properties=self.model.train_properties,
                                      train_results=all_res,
                                      train_set_info=func_set_info_dict(
                                          train_set),
                                      dev_set_info=func_set_info_dict(test_set))
        self.logger.stop_session()
        logs_df = self.logger.get_all_logs_dataframe()
        return logs_df[logs_df['SESS_NAME'] == sess_name]
    # def eval input:self
    # dev_set = [tree, tree, tree]
    #

    def eval(self, dev_set, dev_batch_size):
        test_set = ([0, doc] for doc in dev_set)
        return self.test(test_set, dev_batch_size)

    # def test input:
    #test_set = [label,[tree,tree,tree]]
    #
    def test(self, test_set, dev_batch_size):
        return self.model.eval(self.sess, test_set, dev_batch_size)

    # def k-folds_CV input:
    # set: [label,[tree,tree,tree]]
    # folds: number of randomly partitioned equal sized subsamples, default is 10
    # other options described
    def k_folds_CV(self, set, folds=10, epochs=10, dev_batch_size=1,
                   func_set_info_dict=None, **propery_dict):

        assert folds > 1
        def train_dev_chunks(l, n):
            k = 0
            while k < len(l):
                s = int((len(l)-k)/n)
                n = n - 1
                k += s
                yield l[0:k-s]+l[k:], l[k-s:k]

        random.shuffle(set)
        sess_name = ('%d-folds_CV_' % folds) +  \
                    datetime.datetime.now().strftime('%m_%d_%H_%M')
        for train_s, dev_s in train_dev_chunks(set, folds):
            self.train(train_set=train_s, dev_set=dev_s, epochs=epochs,
                       dev_batch_size=dev_batch_size,
                       func_set_info_dict=func_set_info_dict, save=False,
                       sess_name=sess_name, **propery_dict)
        logs_df = self.logger.get_all_logs_dataframe()
        return logs_df[logs_df['SESS_NAME'] == sess_name]

    def get_all_logs_dataframe(self):
        return self.logger.get_all_logs_dataframe()

    def get_all_logs(self):
        return self.get_all_logs()

    def save_logs(self, path, append=True):
        self.logger.save_logs(path, append)

    # set = [[label1, [first sent tree, second sent tree, ...],
    #        [label2, [first sent tree, second sent tree, ...], ...]]]
    def get_set_labels_distribution(self, set):
        # {label1: count1, label2 : count 2, ..}
        return dict(zip(*np.unique(list(map(itemgetter(0), set)),return_counts=True)))

    def get_set_info(self, set):
        d = {'LABELS_DISTRIBUTION': self.get_set_labels_distribution(set),
             'SET_LEN': len(set)}
        return d

    def restore_model(self, path_to_model, restore_embedding=True,
                                           restore_tree_lstm_cell=True,
                                           restore_sent_lstm=True,
                                           restore_output_layer=True):
        self.model.tree_lstm.restore(self.sess, path_to_model,
            restore_embedding=restore_embedding,
            restore_tree_lstm_cell=restore_tree_lstm_cell)
        self.model.restore(self.sess, path_to_model,
            restore_sent_lstm=restore_sent_lstm,
            restore_output_layer=restore_output_layer)
