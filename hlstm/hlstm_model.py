import tensorflow as tf
import tensorflow_fold as td
import time
import tempfile
import os
import datetime


class HLSTMModel:

    def __init__(self, sess, tree_lstm,
                 sent_lstm_num_units, num_classes):
        self.sess = sess
        self.tree_lstm = tree_lstm
        self.sent_lstm_num_units = sent_lstm_num_units
        self.save_dir = ''
        self.compile_model(num_classes)

    def sent_cell(self):
        return td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(
            num_units=self.sent_lstm_num_units), 'sent_cell')

    def sent_lstm(self):
        return (td.Map(self.tree_lstm.tree_lstm()
                       >> td.Concat()) >> td.RNN(self.sent_cell()))

    def output_layer(self, num_classes):
        return td.FC(num_classes, activation=None, name='output_layer')

    def linearLSTM_over_TreeLstm(self, num_classes):
        return (td.Scalar('int32'), self.sent_lstm() >> td.GetItem(1)
                >> td.GetItem(0) >> self.output_layer(num_classes)) \
            >> self.set_metrics()

    def tf_node_loss(self, logits, labels):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

    def tf_fine_grained_hits(self, logits, labels):
        predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        return tf.cast(tf.equal(predictions, labels), tf.float64)

    def set_metrics(self, train=True):
        """A block that adds metrics for loss and hits; output is the LSTM state."""
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

    def compile_model(self, num_classes):
        self.num_classes = num_classes
        self.model = self.linearLSTM_over_TreeLstm(num_classes)
        self.tree_lstm.resolve_subtree()  # to finish recursive declaration
        self.compiler = td.Compiler.create(self.model)
        print('input type: %s' % self.model.input_type)
        print('output type: %s' % self.model.output_type)

    def prepare_training(self,
                         LEARNING_RATE=0.05,
                         KEEP_PROB=0.75,
                         EMBEDDING_LEARNING_RATE_FACTOR=0.1,
                         BATCH_SIZE=100,
                         init=True,
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
        self.opt = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = self.prepare_word_embedding_gradients()
        if init == True:
            self.sess.run(tf.global_variables_initializer())

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
                     dev_batch_size=1, save=True, save_dir=''):
        self.init_train_set(train_set)
        if dev_set:
            self.init_dev_set(dev_set, dev_batch_size=dev_batch_size)
        all_res = {'EPOCH': [], 'SAVE': [], 'TRAIN_LOSS': [],
                   'TIME_TRAIN_EPOCH_S': [], 'DEV_METRICS': [], 'DEV_EVAL_TIME_S': []}
        for epoch, shuffled in enumerate(td.epochs(self.train_inputs, epochs), 1):
            print('Start epoch ', epoch)
            epoch_res = {key: None for key in all_res.keys()}
            epoch_res['EPOCH'] = epoch
            epoch_res['TRAIN_LOSS'], epoch_res[
                'TIME_TRAIN_EPOCH_S'] = self.train_epoch(shuffled)
            print('Training took ', epoch_res['TIME_TRAIN_EPOCH_S'], 's.')
            checkpoint_path = None
            if save:
                checkpoint_path = self.save_model(save_dir=save_dir)

            if dev_set:
                dev_output, epoch_res['DEV_METRICS'], epoch_res[
                    'DEV_EVAL_TIME_S'] = self.eval_feed_dict()
                print('Evaluation took ', epoch_res['DEV_EVAL_TIME_S'], 's.')

            for key in all_res:
                all_res[key].append(epoch_res[key])
            dev_accuracy = ['%s: %.2f' % (k, v) for k, v in sorted(
                epoch_res['DEV_METRICS'].items())]
            print('epoch:%4d, train_loss: %.3e, dev_accuracy:  %s\n' %
                  (epoch_res['EPOCH'], epoch_res['TRAIN_LOSS'], ' '.join(dev_accuracy)))
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

    @property
    def model_properties(self):
        property = {
            'NUM_CLASSES': self.num_classes,
            'TREE_LSTM_NUM_UNITS': self.tree_lstm.tree_lstm_num_units,
            'SENT_LSTM_NUM_UNITS': self.sent_lstm_num_units,
            'VOCABULARY_LEN': len(self.tree_lstm.vocab),
            'WEIGHTS_SHAPE': self.tree_lstm.weights.shape
        }
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

    def save_model(self, save_dir='', file_name='', global_step=1):
        try:
            if not isinstance(self.saver, tf.train.Saver):
                self.saver = tf.train.Saver()
        except AttributeError:
            self.saver = tf.train.Saver()
        if save_dir:
            self.save_dir = save_dir
        elif not self.save_dir:
            self.save_dir = tempfile.mkdtemp()
        file_name += datetime.datetime.now().strftime('%m_%d_%H_%M')
        path = os.path.join(self.save_dir, file_name)
        checkpoint_path = self.saver.save(
            self.sess, path, global_step=global_step)
        print('Model saved to %s' % checkpoint_path)
        return checkpoint_path

    def restore_model(self, path_to_model):
        try:
            if not isinstance(self.compiler, td.blocks.block_compiler.Compiler):
                print("First you should to compile model.")
                return
        except AttributeError:
            print("First you should to compile model.")
            return

        try:
            self.saver.restore(self.sess, path_to_model)
        except AttributeError:
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, path_to_model)
            print("model restored from: %s" % path_to_model)
