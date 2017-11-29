import copy
import random


class HLSTMInterface:

    def __init__(self, model, logger=Logger()):
        self.model = model
        self.logger = logger

    def get_set_info(self, set):
        d = {'LABELS_DISTRIBUTION': self.get_set_labels_distribution(set),
             'SET_LEN': len(set)}
        return d

    # def train input:
    #   train_set and test_set: [label, [tree, tree, tree, tree]]

    def train(self, train_set, test_set=None, epochs=10, dev_batch_size=1, save=True,
              save_model_dir='', sess_name='', func_set_info_dict=self.get_set_info,
              **propery_dict):
        if not model.is_compiled:
            return -1
        if not sess_name:
            sess_name = 'train_' + datetime.datetime.now().strftime('%m_%d_%H_%M')
        self.logger.start_session(sess_name)
        self.model.prepare_training(propery_dict)
        train_epochs = self.model.train_epochs(train_set, test_set,
                                               dev_batch_size, epochs, dev_batch_size, save, save_model_dir)
        for epoch_res, all_res in train_epochs:
            dev_accuracy = ['%s: %.2f' % (k, v) for k, v in sorted(
                epoch_res['DEV_METRICS'].items())]
            print('epoch:%4d, train_loss: %.3e, dev_accuracy:  %s\n' %
                  (epoch, train_loss, ' '.join(dev_accuracy)))
        self.logger.record_train_logs(model_properies=self.model.model_properties,
            train_properties=self.model.train_properties, train_results=all_res,
            train_set_info=func_set_info(train_set),
            dev_set_info=func_set_info(test_set))
        self.loger.stop_session()
        logs_df = self.logger.get_logs_dataframe()
        return logs_df[logs_df['SESS_NAME'] = sess_name]
    # def eval input:
    # dev_set = [tree, tree, tree]
    #
    def eval(self, dev_set, dev_batch_size):
        test_set = ([0, doc] for doc in dev_set)
        return self.test(trst_set, dev_batch_size)

    # def test input:
    #test_set = [label,[tree,tree,tree]]
    #
    def test(self, test_set, dev_batch_size):
        return self.model.eval(test_set, dev_batch_size)

    # def k-folds_CV input:
    # set: [label,[tree,tree,tree]]
    # folds: number of randomly partitioned equal sized subsamples, default is 10
    # other options described
    def k-folds_CV(self, set, folds=10, epochs=10, dev_batch_size=1,
                   func_set_info_dict=self.get_set_info, **propery_dict):
        assert folds > 1
        def train_dev_chunks(l, n):
            k = 0
            while k < len(l):
                s = int((len(l)-k)/n)
                n = n - 1
                k += s
                yield l[0:k-s]+l[k:], l[k-s:k]

        random.shuffle(set)
        sess_name = '%d-folds_CV_'+datetime.datetime.now().strftime('%m_%d_%H_%M')
        for train_s, dev_s in train_dev_chunks(set, folds):
            self.train(train_set=train_s, dev_set=dev_s, epochs=epochs,
                       dev_batch_size=dev_batch_size,
                       func_set_info_dict=func_set_info_dict, save=False,
                       sess_name=sess_name, **propery_dict)
        logs_df = self.logger.get_all_logs_dataframe()
        return logs_df[logs_df['SESS_NAME'] = sess_name]

    def get_all_logs_dataframe(self):
        return self.logger.get_all_logs_dataframe()

    def get_all_logs(self):
        retun self.get_all_logs()

    def save_logs(self, path, append=True):
        self.logger.save_logs(path, append)

    # set = [[label1, [first sent tree, second sent tree, ...],
    def get_set_labels_distribution(self, set):
                                                #       [label2, [first sent tree, second sent tree, ...], ...]]]

        # {label1: count1, label2 : count 2, ..}
        return dict(zip(*np.unique(list(map(itemgetter(0), set)), return_counts=True)))
