import random
import re
from string import punctuation

from nltk.tokenize.sexpr import sexpr_tokenize
import settings.STOP_WORDS


class TreeBinarizer():

    def __init__(self, myvocab, replace_dict):
        self.vocab = myvocab
        self.replace_dict = replace_dict
        self.STOP_WORDS = settings.STOP_WORDS

    def split_to_two_lists(self, lis):
        random.shuffle(lis)
        e = random.randint(1, len(lis)-1)
        return lis[0:e], lis[e:]

    def parse_subtree(self, s):
        try:
            root, children = s[1:-1].lstrip().split(' ', 1)
        except ValueError:
            root = ''
            children = s[1:-1].lstrip()
        return (root.strip(punctuation+' '),
                sexpr_tokenize(children.strip()))

    def is_noize(self, s):
        st = self.parse_subtree(s)[0]
        return ((st in punctuation) or (st in self.STOP_WORDS) or
                (st.strip() == '') or (not st in self.vocab))

    def replace(self, s):
        # replace with value if pattern(key) in word, if there are some several
        # patterns, replace with random
        for key, value in self.replace_dict:
            if re.seach(key, s):
                s = value
        return s

    def remove_noize(self, l):
        while True:
            # print(l)
            good = []
            bad = []
            for x in l:
                bad.append(x) if self.is_noize(x,) else good.append(x)
            if not bad:
                return good
            else:
                for b in bad:
                    good += self.parse_subtree(b)[1]
                l = good

    def to_binary_subtree(self, par, is_root=False):
        # print(par)
        if is_root and self.is_noize(par[0]):
            children = self.remove_noize(self.parse_subtree(par[0])[1])
            if not children:
                return ' '
            else:
                return self.to_binary_subtree(children)

        if len(par) == 0:
            raise RuntimeError("len(par) == 0")
        if len(par) == 1:
            children = self.remove_noize(self.parse_subtree(par[0])[1])
            if not children:
                return self.replace(self.parse_subtree(par[0])[0])
            else:
                return '( ' + self.replace(self.parse_subtree(par[0])[0]) + \
                    ' ) ( ' + self.to_binary_subtree(children,) + ' )'
        if len(par) > 1:
            return '( ' + ' ) ( '.join([self.to_binary_subtree(p,)
                                        for p in self.split_to_two_lists(par)]) + ' )'

    def to_binary_tree(self, tree):
        return '( '+self.to_binary_subtree([tree], is_root=True) + ')'
