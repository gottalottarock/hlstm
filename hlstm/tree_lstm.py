import tensorflow as tf
import tensorflow_fold as td
import numpy as np
from tensorflow.contrib.framework import list_variables
from nltk.tokenize.sexpr import sexpr_tokenize

from .tree_lstm_cell import BinaryTreeLSTMCell
from .tree_binarizer import TreeBinarizer
from .exceptions import VariableNotFoundException


class BinaryTreeLSTM:

    _word_embedding_default_scope_name = 'word_embedding'
    _tree_lstm_cell_default_scope_name = 'tree_lstm_cell'

    @classmethod
    def get_default_scope_names(cls):
        l = []
        for key, value in cls.__dict__.items():
            if key.endswith('_default_scope_name') and isinstance(value, str):
                l.append(value)
        return l

    def __init__(self, weights, vocab, tree_lstm_num_units, tree_binarizer=None):
        if not tree_binarizer:
            tree_binarizer = TreeBinarizer(vocab, dict())
        self.tree_binarizer = tree_binarizer
        self.tree_lstm_keep_prob_ph = tf.placeholder_with_default(1.0, [])
        self.tree_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                BinaryTreeLSTMCell(tree_lstm_num_units,
                                   self.tree_lstm_keep_prob_ph),
                self.tree_lstm_keep_prob_ph, self.tree_lstm_keep_prob_ph),
                name_or_scope=self._tree_lstm_cell_default_scope_name)
        self.word_embedding = td.Embedding(
            *weights.shape, initializer=weights, name=self._word_embedding_default_scope_name)
        self.embed_subtree = td.ForwardDeclaration(name='embed_subtree')
        self.vocab = vocab

    @classmethod
    def init_from_file(cls, filename, vocab):
        var_shape = dict(list_variables(filename))
        try:
            embed_shape = var_shape['word_embedding/weights']
        except KeyError:
            raise VariableNotFoundException(variable='word_embedding/weights',
                                            where='file %s' % filename,
                                            msg='Try to initialize manually.')
        vocab_len, embed_len = embed_shape
        vocab_len = vocab_len - 1
        try:
            tree_lstm_weights_shape = var_shape[
                'tree_lstm_cell/fully_connected/weights']
        except KeyError:
            raise VariableNotFoundException(variable='tree_lstm_cell/fully_connected/weights',
                                            where='file %s' % filename,
                                            msg='Try to initialize manually.')
        tree_lstm_num_units = int(tree_lstm_weights_shape[1]/5)
        assert tree_lstm_weights_shape[0] == embed_len + tree_lstm_num_units*2
        if not vocab_len == len(vocab):
            raise RuntimeError('Vocab used with saved model had a different size.\n \
                                Try to initialize manually\n \
                                Used vocab len: %d'% vocab_len)
        weights = np.zeros(embed_shape, dtype = np.float32)
        tree_lstm = cls(weights, vocab, tree_lstm_num_units)
        return tree_lstm

    def logits_and_state(self):
        """Creates a block that goes from tokens to (logits, state) tuples."""
        unknown_idx = len(self.vocab)

        def lookup_word(word): return self.vocab.get(word, unknown_idx)

        #(GetItem(key) >> block).eval(inp) => block.eval(inp[key])
        # InputTransform(funk): A Python function, lifted to a block.
        # Scalar - input to scalar
        word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                    td.Scalar('int32') >> self.word_embedding)
        #
        pair2vec = (self.embed_subtree(), self.embed_subtree())

        # Trees are binary, so the tree layer takes two states as its
        # input_state.
        zero_state = td.Zeros((self.tree_lstm_cell.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        # AllOf(a, b, c).eval(inp) => (a.eval(inp), b.eval(inp), c.eval(inp))
        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)
        # OneOf(func, [(key, block),(key,block)])) where funk(input) => key and
        # OneOf returns one of blocks
        tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

        return tree2vec >> self.tree_lstm_cell

    def tree_transform(self, p):
        try:
            b_tree = self.tree_binarizer.to_binary_tree(p)
        except RecursionError:
            b_tree = '(()())'
        return b_tree

    def tokenize(self, s):
        if not s[1:-1].strip():
            return ['']
        return sexpr_tokenize(s[1:-1].strip())

    def embed_tree(self):
        return td.InputTransform(self.tokenize) >> self.logits_and_state() \
            >> td.GetItem(1)

    def tree_lstm(self):
        return td.InputTransform(self.tree_transform) >> self.embed_tree()

    def resolve_subtree(self):
        self.embed_subtree.resolve_to(self.embed_tree())

    def prepare_var_dict_for_saver(self, embedding,
                                   tree_lstm_cell):
        def save_name(name, def_pref):
            name = def_pref + '/' + name.split('/', 1)[1]
            name = name.rsplit(':',1)[0]
            return name

        var_dict = dict()
        if embedding:
            for var in self.embedding_variables:
                var_dict[
                    save_name(var.name, self._word_embedding_default_scope_name)] = var

        if tree_lstm_cell:
            for var in self.tree_lstm_variables:
                var_dict[
                    save_name(var.name, self._tree_lstm_cell_default_scope_name)] = var
        return var_dict

    def prepare_restorer(self, embedding=True, tree_lstm_cell=True):
        var_dict = self.prepare_var_dict_for_saver(embedding=embedding,
                                                   tree_lstm_cell=tree_lstm_cell)
        saver = tf.train.Saver(var_list=var_dict, max_to_keep=None)
        return var_dict, saver

    def restore(self, sess, path_to_model, restore_embedding=True,
                restore_tree_lstm_cell=True):
        var_dict, restorer = self.prepare_restorer(embedding=restore_embedding,
                                                tree_lstm_cell=restore_tree_lstm_cell)
        restorer.restore(sess, path_to_model)

    @property
    def properties(self):
        property = {'TREE_LSTM_NUM_UNITS': self.tree_lstm_num_units,
                    'WEIGHTS_SHAPE': self.weights_shape,
                    'VOCABULARY_LEN': self.vocab_len}
        return property

    @property
    def tree_lstm_num_units(self):
        return self.tree_lstm_cell.state_size[0]

    @property
    def weights_shape(self):
        return (self.word_embedding.num_buckets, self.word_embedding.num_units_out)

    @property
    def vocab_len(self):
        return len(self.vocab)

    @property
    def tree_lstm_name(self):
        return self.tree_lstm_cell.name

    @property
    def embedding_name(self):
        return self.word_embedding.name

    @property
    def variables_names(self):
        return [self.embedding_name,
                self.tree_lstm_name]

    @property
    def tree_lstm_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.tree_lstm_name)

    @property
    def embedding_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.embedding_name)

    @property
    def variables(self):
        return self.tree_lstm_variables + self.embedding_variables
