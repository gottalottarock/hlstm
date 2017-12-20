import tensorflow as tf
import tensorflow_fold as td
import time
import tempfile
import os
import datetime
from tensorflow.contrib.framework import list_variables

from .exceptions import VariableNotFoundException

class HLSTMModel:

    _sent_lstm_default_scope_name = 'sent_cell'

    _output_layer_default_scope_name = 'output_layer'

    _optimizer_scope_name = 'optimizer'
    
    @classmethod
    def get_default_scope_names(cls):
        l = []
        for key, value in cls.__dict__.items():
            if key.endswith('_default_scope_name') and isinstance(value, str):
                l.append(value)
        return l

    def __init__(self, sess, tree_lstm, sent_lstm_num_units, num_classes):
        self.sess = sess
        self.tree_lstm = tree_lstm
        self.save_dir = ''
        self.compile_model(num_classes, sent_lstm_num_units)

    @classmethod
    def init_from_file(cls,filename,sess,tree_lstm):
        var_shape = dict(list_variables(filename))
        try:
            output_layer_weights_shape = var_shape['output_layer/weights']
        except KeyError:
            raise VariableNotFoundException(variable='output_layer/weights',
                                            where='file %s' % filename,
                                            msg='Try to initialize manually.')
        try:
            sent_cell_weights_shape = var_shape['sent_cell/weights:0']
        except KeyError:
            raise VariableNotFoundException(variable='sent_cell/weights:0',
                                            where='file %s' % filename,
                                            msg='Try to initialize manually.')
        num_classes = output_layer_weights_shape[1]
        sent_lstm_num_units = sent_cell_weights_shape[1]/4
        assert sent_lstm_num_units == output_layer_weights_shape[0]
        if (sent_cell_weights_shape[0] ==
            sent_lstm_num_units + tree_lstm.tree_lstm_num_units*2):
            raise RuntimeError('Saved model and tree lstm not compatible.')
        model = cls(sess, tree_lstm, sent_lstm_num_units, num_classes)
        return model



    def linearLSTM_over_TreeLstm(self, num_classes, sent_lstm_num_units):
        self.sent_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(
            num_units=sent_lstm_num_units), name_or_scope = self._sent_lstm_default_scope_name)
        sent_lstm = (td.Map(self.tree_lstm.tree_lstm()
                            >> td.Concat()) >> td.RNN(self.sent_cell))
        self.output_layer = td.FC(
            num_classes, activation=None, name=self._output_layer_default_scope_name)
        return (td.Scalar('int32'), sent_lstm >> td.GetItem(1)
                >> td.GetItem(0) >> self.output_layer) \
            >> self.set_metrics()

    def tf_node_loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

    def tf_fine_grained_hits(self, logits, labels):
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        return tf.cast(tf.equal(predictions, labels), tf.float64)

    def set_metrics(self, train=True):
        """A block that adds metrics for loss and hits;
           output is the LSTM state."""
        c = td.Composition(
            name='predict')
        with c.scope():
            # destructure the input; (labels, logits)
            labels = c.input[0]
            logits = c.input[1]

            # calculate loss
            loss = td.Function(self.tf_node_loss)
            td.Metric('root_loss').reads(loss.reads(logits, labels))

            hits = td.Function(self.tf_fine_grained_hits)
            td.Metric('root_hits').reads(hits.reads(logits, labels))

            c.output.reads(logits)
        return c

    def compile_model(self, num_classes, sent_lstm_num_units):
        self.model = self.linearLSTM_over_TreeLstm(
            num_classes, sent_lstm_num_units)
        self.tree_lstm.resolve_subtree()  # to finish recursive declaration
        self.compiler = td.Compiler.create(self.model)
        print('input type: %s' % self.model.input_type)
        print('output type: %s' % self.model.output_type)

    def prepare_training(self,
                         LEARNING_RATE=0.005,
                         KEEP_PROB=0.75,
                         EMBEDDING_LEARNING_RATE_FACTOR=0.1,
                         BATCH_SIZE=100,
                         init_model_variables=True,
                         **extras):
        self.LEARNING_RATE = LEARNING_RATE
        self.KEEP_PROB = KEEP_PROB
        self.EMBEDDING_LEARNING_RATE_FACTOR = EMBEDDING_LEARNING_RATE_FACTOR
        self.BATCH_SIZE = BATCH_SIZE
        self.train_feed_dict = {
            self.tree_lstm.tree_lstm_keep_prob_ph: self.KEEP_PROB}
        self.metrics = {k: tf.reduce_mean(
            v) for k, v in self.compiler.metric_tensors.items()}
        self.loss = tf.reduce_sum(self.compiler.metric_tensors['root_loss'])
        # opt = tf.train.AdagradOptimizer(LEARNING_RATE)
        with tf.variable_scope(self._optimizer_scope_name):
            self.opt = tf.train.AdamOptimizer(self.LEARNING_RATE)
            self.train = self.prepare_word_embedding_gradients()
        if init_model_variables:
            self.init_model_variables()
        self.init_optimizer()

    def prepare_word_embedding_gradients(self):
        grads_and_vars = self.opt.compute_gradients(self.loss)
        found = 0
        for i, (grad, var) in enumerate(grads_and_vars):
            if var == self.tree_lstm.word_embedding.weights:
                found += 1
                grad = tf.scalar_mul(self.EMBEDDING_LEARNING_RATE_FACTOR, grad)
                grads_and_vars[i] = (grad, var)
        assert found == 1  # internal consistency check
        self.train_grad = self.opt.apply_gradients(grads_and_vars)

    def train_step(self, batch):
        self.train_feed_dict[self.compiler.loom_input_tensor] = batch
        _, batch_loss = self.sess.run(
            [self.train_grad, self.loss], self.train_feed_dict)
        return batch_loss

    def train_epoch(self, train_input=None):
        if not self.is_compiled:
            return 0, 0
        if not train_input:
            train_input = self.train_inputs
        t = time.time()
        loss = sum(self.train_step(ba)
                   for ba in td.group_by_batches(train_input, self.BATCH_SIZE))
        t = time.time() - t
        return loss, t

    def eval_feed_dict(self):
        if not self.is_compiled:
            return list(), dict(), 0
        t = time.time()
        output, metrics = self.sess.run(
            [self.compiler.output_tensors[0], self.metrics], self.dev_feed_dict)
        t = time.time() - t
        return output, metrics, t

    def init_train_set(self, train_set, **extras):
        self.train_inputs = self.compiler.build_loom_inputs(train_set)
        return train_set

    def init_dev_set(self, dev_set, dev_batch_size=1, **extras):
        self.dev_feed_dict = self.compiler.build_feed_dict(
            dev_set, batch_size=dev_batch_size)
        return dev_set

    def train_epochs(self, train_set, dev_set=None, epochs=10,
                     dev_batch_size=1, save=True, save_dir='',
                     quiet_saver = True):
        self.init_train_set(train_set)
        if dev_set:
            self.init_dev_set(dev_set, dev_batch_size=dev_batch_size)
        all_res = {'EPOCH': [], 'SAVE': [], 'TRAIN_LOSS': [],
                   'TIME_TRAIN_EPOCH_S': [], 'DEV_METRICS': [],
                   'DEV_EVAL_TIME_S': []}
        for epoch, shuffled in enumerate(td.epochs(self.train_inputs, epochs), 1):
            print('Start epoch ', epoch)
            epoch_res = {key: None for key in all_res.keys()}
            epoch_res['EPOCH'] = epoch
            epoch_res['TRAIN_LOSS'], epoch_res[
                'TIME_TRAIN_EPOCH_S'] = self.train_epoch(shuffled)
            print('Training took %.2f s.' % epoch_res['TIME_TRAIN_EPOCH_S'])
            checkpoint_path = None

            if dev_set:
                dev_output, epoch_res['DEV_METRICS'], epoch_res[
                    'DEV_EVAL_TIME_S'] = self.eval_feed_dict()
                print('Evaluation took %.2f s.' % epoch_res['DEV_EVAL_TIME_S'])

            dev_accuracy = ['%s: %.2f' % (k, v) for k, v in sorted(
                epoch_res['DEV_METRICS'].items())]
            if save:
                checkpoint_path = self.save_model(
                    save_dir=save_dir, global_step=epoch, quiet = quiet_saver)
                epoch_res['SAVE'] = checkpoint_path
            else:
                epoch_res['SAVE'] = ''
            print('epoch:%4d, train_loss: %.3e, dev_accuracy:  %s\n' %
                  (epoch_res['EPOCH'], epoch_res['TRAIN_LOSS'],
                   ' '.join(dev_accuracy)))
            for key in all_res:
                all_res[key].append(epoch_res[key])
            yield epoch_res, all_res

    def eval(self, dev_set, dev_batch_size=1):
        self.init_dev_set(dev_set, dev_batch_size=dev_batch_size)
        return self.eval_feed_dict()

    @property
    def is_compiled(self):
        try:
            if not isinstance(self.compiler, td.blocks.block_compiler.Compiler):
                print("First you should to compile model.")
                return False
        except AttributeError:
            print("First you should to compile model.")
            return False
        return True

    def save_model(self, save_dir='', file_name='', global_step=1,
                   save_embedding=True, save_tree_lstm_cell=True,
                   save_sent_lstm=True, save_output_layer=True, quiet=False):
        var_dict, saver = self.prepare_saver(embedding=save_embedding,
                                             tree_lstm_cell=save_tree_lstm_cell,
                                             sent_lstm=save_sent_lstm,
                                             output_layer=save_output_layer)
        if save_dir:
            self.save_dir = save_dir
        elif not self.save_dir:
            self.save_dir = tempfile.mkdtemp()
        file_name += datetime.datetime.now().strftime('%m_%d_%H_%M')
        path = os.path.join(self.save_dir, file_name)

        checkpoint_path = saver.save(
            self.sess, path, global_step=global_step)
        print('Model saved to %s' % checkpoint_path)
        if not quiet:
            print("Saved variables:")
            print('\n'.join(sorted(var_dict.keys())))
        return checkpoint_path

    def restore(self, path_to_model, restore_sent_lstm=True,
                restore_output_layer=True):
        var_dict, restorer = self.prepare_restorer(sent_lstm=restore_sent_lstm,
                                             output_layer=restore_output_layer)
        restorer.restore(self.sess, path_to_model)


    def prepare_restorer(self, sent_lstm=True, output_layer=True):
        var_dict = self.prepare_var_dict_for_saver(sent_lstm=sent_lstm,
                                                   output_layer=output_layer)
        restorer = tf.train.Saver(var_list=var_dict, max_to_keep=None)
        return var_dict, restorer

    def prepare_saver(self, embedding=True, tree_lstm_cell=True,
                      sent_lstm=True, output_layer=True):
        tree_lstm_var_dict = self.tree_lstm.prepare_var_dict_for_saver(embedding=embedding,
                                                                       tree_lstm_cell=tree_lstm_cell)
        sent_lstm_var_dict = self.prepare_var_dict_for_saver(sent_lstm=sent_lstm,
                                                             output_layer=output_layer)
        var_dict = dict(tree_lstm_var_dict, **sent_lstm_var_dict)
        saver = tf.train.Saver(var_list=var_dict, max_to_keep=None)
        return var_dict, saver

    def remove_opt_variables(self, var_dict):
        try:
            opt_name = self.opt.get_name()
        except AttributeError:
            return var_dict
        var_dict = {key:v for key,v in var_dict.items()
                    if not key.rsplit('/',1)[1].startswith(opt_name)}
        return var_dict

    def prepare_var_dict_for_saver(self, sent_lstm, output_layer):
        def save_name(name, def_pref):
            return def_pref + '/' + name.split('/', 1)[1]
        var_dict = dict()
        if sent_lstm:
            for var in self.sent_cell_variables:
                var_dict[save_name(var.name, self._sent_lstm_default_scope_name)] = var
        if output_layer:
            for var in self.output_layer_variables:
                var_dict[save_name(var.name, self._output_layer_default_scope_name)] = var
        return var_dict

    def init_model_variables(self):
        self.sess.run(tf.variables_initializer(self.variables))

    def init_optimizer(self):
        self.sess.run(tf.variables_initializer(self.optimizer_variables))

    @property
    def num_classes(self):
        return self.output_layer.output_size

    @property
    def sent_lstm_num_units(self):
        return self.sent_cell.state_size  # ????

    @property
    def model_properties(self):
        property = {
            'NUM_CLASSES': self.num_classes,
            'SENT_LSTM_NUM_UNITS': self.sent_lstm_num_units,  # ??????
        }
        property.update(self.tree_lstm.properties)
        return property

    @property
    def train_properties(self):
        property = dict()
        try:
            property = {
                'LEARNING_RATE': self.LEARNING_RATE,
                'KEEP_PROB': self.KEEP_PROB,
                'EMBEDDING_LEARNING_RATE_FACTOR': self.EMBEDDING_LEARNING_RATE_FACTOR,
                'BATCH_SIZE': self.BATCH_SIZE
            }
        except AttributeError:
            print("Training properties not prepared, return empty dict")
        return property



    @property
    def sent_cell_name(self):
        return self.sent_cell.name

    @property
    def sent_cell_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.sent_cell_name)

    @property
    def output_layer_name(self):
        return self.output_layer.name

    @property
    def output_layer_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.output_layer_name)

    @property
    def variables_names(self):
        return [self.sent_cell_name,
                self.output_layer_name] + self.tree_lstm.variables_names

    @property
    def variables(self):
        vars = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            for name in self.variables_names:
                name = name + '/'
                if var.name.startswith(name):
                    vars.append(var)
        return vars + self.tree_lstm.variables

    @property
    def optimizer_variables(self):
        vars = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if var.name.startswith(self._optimizer_scope_name):
                vars.append(var)
        return vars