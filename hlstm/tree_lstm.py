import tensorflow as tf
import tensorflow_fold as td
from .tree_lstm_cell import BinaryTreeLSTMCell


class BinaryTreeLSTM:

    def __init__(self, weights, vocab, tree_lstm_num_units):
        self.tree_lstm_keep_prob_ph = tf.placeholder_with_default(1.0, [])
        self.tree_lstm_num_units = tree_lstm_num_units
        self.tree_lstm = td.ScopedLayer(
            tf.contrib.rnn.DropoutWrapper(
                BinaryTreeLSTMCell(tree_lstm_num_units,
                                   self.tree_lstm_keep_prob_ph),
                self.tree_lstm_keep_prob_ph, self.tree_lstm_keep_prob_ph),
            name_or_scope='tree_lstm')
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
        zero_state = td.Zeros((self.tree_lstm.state_size,) * 2)
        # Input is a word vector.
        zero_inp = td.Zeros(self.word_embedding.output_type.shape[0])

        # AllOf(a, b, c).eval(inp) => (a.eval(inp), b.eval(inp), c.eval(inp))
        word_case = td.AllOf(word2vec, zero_state)
        pair_case = td.AllOf(zero_inp, pair2vec)
        # OneOf(func, [(key, block),(key,block)])) where funk(input) => key and
        # OneOf returns one of blocks
        tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

        return tree2vec >> self.tree_lstm

    def tokenize(self, s):
        if not s[1:-1].strip():
            return ['']
        return sexpr_tokenize(s[1:-1].strip())

    def embed_tree(self):
        return td.InputTransform(self.tokenize) >> self.logits_and_state() \
            >> td.GetItem(1)

    def resolve_subtree(self):
        self.embed_subtree.resolve_to(self.embed_tree())
