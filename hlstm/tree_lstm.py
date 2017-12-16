import tensorflow as tf
import tensorflow_fold as td
from nltk.tokenize.sexpr import sexpr_tokenize
from .tree_lstm_cell import BinaryTreeLSTMCell
from .tree_binarizer import TreeBinarizer


class BinaryTreeLSTM:

    def __init__(self, sess, weights, vocab, tree_lstm_num_units, tree_binarizer=None):
        if not tree_binarizer:
            tree_binarizer = TreeBinarizer(vocab, dict())
        self.sess = sess
        self.tree_binarizer = tree_binarizer
        self.tree_lstm_keep_prob_ph = tf.placeholder_with_default(1.0, [])
        self.tree_lstm_cell = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                BinaryTreeLSTMCell(tree_lstm_num_units,
                                   self.tree_lstm_keep_prob_ph),
                self.tree_lstm_keep_prob_ph, self.tree_lstm_keep_prob_ph),
            name_or_scope='tree_lstm_cell')
        self.word_embedding = td.Embedding(
            *weights.shape, initializer=weights, name='word_embedding')
        self.embed_subtree = td.ForwardDeclaration(name='embed_subtree')
        self.weights = weights
        self.vocab = vocab

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

    def prepare_var_dict_for_saver(self,embedding,
                                   tree_lstm_cell):
        def save_name(name, def_pref):
            return def_pref + '/' + name.split('/',1)[1]

        var_dict = dict()
        if embedding:
            for var in self.embedding_variables:
                var_dict[save_name(var.name, 'word_embedding' )] = var

        if tree_lstm_cell:
            for var in self.tree_lstm_variables:
                var_dict[save_name(var.name, 'tree_lstm_cell')] = var
        return var_dict

    @property
    def properties(self):
        property = {'TREE_LSTM_NUM_UNITS': self.tree_lstm_num_units,
                    'WEIGHTS_SHAPE': self.weights_shape,
                    'VOCABULARY_LEN': self.vocab_len}
        return property
    
    @property
    def tree_lstm_num_units(self):
        return self.tree_lstm_cell.state_size

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
                                 scope= self.tree_lstm_name)

    @property
    def embedding_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.embedding_name)

    @property
    def variables(self):
        return self.tree_lstm_variables + self.embedding_variables
