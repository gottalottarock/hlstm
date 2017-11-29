import random
import re
from string import punctuation

from nltk.tokenize.sexpr import sexpr_tokenize


class TreeBinarizer():

    def __init__(self, myvocab, replace_dict):
        self.vocab = myvocab
        self.replace_dict = replace_dict
        self.STOP_WORDS = {'around', 'namely', 'against', 'front', 'only',
                           'third', 'very', 'however', 'again', 'top', 'until', 'hundred',
                           'between', 'towards', 'meanwhile', 'always', 'mostly', 'should',
                           'regarding', 'cannot', 'such', 'where', 'a', 'own', 'put',
                           'wherein', 'forty', 'whereupon', 'yourself', 'besides', 'alone',
                           'can', 'there', 'herself', 'much', 'never', 'whether', 'with',
                           'therein', 'his', 'into', 'has', 'rather', 'nowhere', 'although',
                           'itself', 'those', 'throughout', 'its', 'in', 'less', 'then',
                           'whom', 'is', 'therefore', 'this', 'yourselves', 'both', 'almost',
                           'four', 'inc', 'anyhow', 'hereby', 'ten', 'ca', 'ourselves', 'the',
                           'these', 'who', 'becoming', 'thereafter', 'could', 'been', 'that',
                           'us', 'take', 'because', 'thence', 'even', 'together', 'whither',
                           'etc', 'several', 'your', 'give', 'side', 'we', 'back', 'become',
                           'being', 'empty', 'among', 'afterwards', 'be', 'across', 'already',
                           'he', 'next', 'whoever', 'unless', 'last', 'used', 'full', 'what',
                           'above', 'their', 'latterly', 'also', 'did', 'noone', 'else',
                           'elsewhere', 'name', 'former', 'by', 'fifteen', 'beforehand',
                           'serious', 'have', 'seemed', 'everywhere', 'five', 'six', 'each',
                           'as', 'so', 'anything', 'someone', 'see', 'one', 'himself',
                           'about', 'must', 'ever', 'during', 'nobody', 'whose', 'on', 'from',
                           'two', 'was', 'onto', 'yet', 'since', 'am', 'many', 'whereafter',
                           'using', 'though', 'thus', 'whole', 'all', 'our', 're', 'per',
                           'once', 'it', 'might', 'over', 'became', 'seems', 'to', 'which',
                           'formerly', 'just', 'ours', 'below', 'amongst', 'would', 'due',
                           'now', 'behind', 'really', 'further', 'becomes', 'make', 'enough',
                           'for', 'please', 'amount', 'fifty', 'if', 'along', 'myself',
                           'and', 'were', 'after', 'move', 'still', 'well', 'him', 'twelve',
                           'neither', 'thereby', 'wherever', 'they', 'anyone', 'me', 'few',
                           'more', 'sixty', 'latter', 'through', 'keep', 'seeming',
                           'anywhere', 'of', 'when', 'you', 'doing', 'bottom', 'either',
                           'does', 'off', 'somehow', 'others', 'at', 'mine', 'beside',
                           'down', 'indeed', 'eleven', 'everyone', 'whenever', 'often',
                           'nevertheless', 'an', 'any', 'while', 'thru', 'go', 'will',
                           'except', 'some', 'too', 'them', 'her', 'three', 'most',
                           'nothing', 'or', 'show', 'do', 'twenty', 'nor', 'had', 'various',
                           'least', 'within', 'she', 'why', 'say', 'without', 'somewhere',
                           'yours', 'anyway', 'i', 'toward', 'something', 'before', 'beyond',
                           'themselves', 'via', 'eight', 'whereby', 'than', 'thereupon',
                           'my', 'sometime', 'get', 'quite', 'made', 'no', 'whereas', 'may',
                           'part', 'herein', 'sometimes', 'out', 'another', 'whatever',
                           'here', 'moreover', 'upon', 'but', 'call', 'same', 'perhaps',
                           'seem', 'under', 'are', 'done', 'how', 'not', 'otherwise', 'hers',
                           'none', 'hereupon', 'other', 'hence', 'up', 'whence', 'first',
                           'every', 'nine', 'hereafter', 'everything'}

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
