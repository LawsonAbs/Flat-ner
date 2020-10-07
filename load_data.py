from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import numpy as np
import fitlog
import pickle
import os
from fastNLP import cache_results
# from fastNLP.embeddings import StaticEmbedding
from fastNLP_module import StaticEmbedding


@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_ontonotes4ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,train_clip=False,
                       char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path,'train.char.bmes{}'.format('_clip' if train_clip else ''))
    dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']])
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/resume_ner',_refresh=False)
def load_resume_ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,
                    char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path,'train.char.bmes')
    dev_path = os.path.join(path,'dev.char.bmes')
    test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()

    datasets['test'] = test_bundle.datasets['train']
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    
    print(datasets)
    # 作用？
    # apply_field()
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings


@cache_results(_cache_fp='need_to_defined_fp',_refresh=False)
def equip_chinese_ner_with_skip(datasets,vocabs,embeddings,w_list,word_embedding_path=None,
                                word_min_freq=1,only_train_min_freq=0):
    from utils_ import Trie,get_skip_path
    from functools import partial
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)

    # for k,v in datasets.items():
    #     v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','skips')

    def skips2skips_l2r(chars,w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars,w_trie=w_trie)


        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[e].append([s,w])

        return result

    def skips2skips_r2l(chars,w_trie):
        '''

        :param lexicons: list[[int,int,str]]
        :return: skips_l2r
        '''
        # print(lexicons)
        # print('******')

        lexicons = get_skip_path(chars,w_trie=w_trie)


        # max_len = max(list(map(lambda x:max(x[:2]),lexicons)))+1 if len(lexicons) != 0 else 0

        result = [[] for _ in range(len(chars))]

        for lex in lexicons:
            s = lex[0]
            e = lex[1]
            w = lex[2]

            result[s].append([e,w])

        return result

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_l2r,w_trie=w_trie),'chars','skips_l2r')

    for k,v in datasets.items():
        v.apply_field(partial(skips2skips_r2l,w_trie=w_trie),'chars','skips_r2l')

    # print(v['skips_l2r'][0])
    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab
    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_l2r','skips_l2r_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_l2r', 'skips_l2r_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:[ list(map(lambda x:x[0],p)) for p in x],'skips_r2l','skips_r2l_source')
        v.apply_field(lambda x:[ list(map(lambda x:x[1],p)) for p in x], 'skips_r2l', 'skips_r2l_word')

    for k,v in datasets.items():
        v.apply_field(lambda x:list(map(len,x)), 'skips_l2r_word', 'lexicon_count')
        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_l2r_word',new_field_name='skips_l2r_word')

        v.apply_field(lambda x:list(map(len,x)), 'skips_r2l_word', 'lexicon_count_back')

        v.apply_field(lambda x:
                      list(map(lambda y:
                               list(map(lambda z:word_vocab.to_index(z),y)),x)),
                      'skips_r2l_word',new_field_name='skips_r2l_word')





    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    vocabs['char'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                              field_name='target', new_field_name='target')

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list',_refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path,drop_characters=True):
    f = open(embedding_path,'r')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x:len(x) != 1, w_list))

    return w_list






@cache_results(_cache_fp='cache/ontonotes4ner',_refresh=False)
def load_toy_ner(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,train_clip=False):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path,'toy_train.bmes')
    dev_path = os.path.join(path,'toy_dev.bmes')
    test_path = os.path.join(path,'toy_test.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None,unknown=None)
    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['dev'],datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['dev'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings



@cache_results(_cache_fp='cache/msraner1',_refresh=False)
def load_msra_ner_1(path,char_embedding_path=None,bigram_embedding_path=None,index_token=True,train_clip=False,
                              char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    if train_clip:
        train_path = os.path.join(path, 'train_dev.char.bmes_clip1')
        test_path = os.path.join(path, 'test.char.bmes_clip1')
    else:
        train_path = os.path.join(path,'train_dev.char.bmes')
        test_path = os.path.join(path,'test.char.bmes')

    loader = ConllLoader(['chars','target'])
    train_bundle = loader.load(train_path)
    test_bundle = loader.load(test_path)


    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']


    datasets['train'].apply_field(get_bigrams,field_name='chars',new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')



    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()
    print(datasets.keys())
    # print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'],field_name='chars',
                            no_create_entry_dataset=[datasets['test']] )
    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='chars',new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'],datasets['test'],
                                 field_name='target',new_field_name='target')

    vocabs = {}
    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab
    vocabs['bigram'] = bigram_vocab
    vocabs['label'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab,char_embedding_path,word_dropout=0.01,
                                         min_freq=char_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab,bigram_embedding_path,word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets,vocabs,embeddings


# 如果加了@cache_results 这个参数，那么在debug 的时候，可能是不会进到这个函数中
#@cache_results(_cache_fp='cache/weiboNER_uni+bi', _refresh=False)
def load_weibo_ner(path,unigram_embedding_path=None,bigram_embedding_path=None,index_token=True,
                   char_min_freq=1,bigram_min_freq=1,only_train_min_freq=0,char_word_dropout=0.01):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    loader = ConllLoader(['chars','target'])
    train_path = os.path.join(path,'weiboNER_2nd_conll.train_deseg')
    dev_path = os.path.join(path, 'weiboNER_2nd_conll.dev_deseg')
    test_path = os.path.join(path, 'weiboNER_2nd_conll.test_deseg')

    paths = {}
    paths['train'] = train_path
    paths['dev'] = dev_path
    paths['test'] = test_path

    datasets = {}  # 字典！！！

    for k,v in paths.items():
        bundle = loader.load(v)
        datasets[k] = bundle.datasets['train']
    
    trainData = datasets['train']
    
    print(type(trainData))  # <class 'fastNLP.core.dataset.DataSet'>
    print(len(trainData))   # 1350
    print(trainData)
    """ datasets['train'] 中的数据长成下面这样，
        +-----------------------------------------------------------+-----------------------------------------------------------+
        | chars                                                     | target                                                    |
        +-----------------------------------------------------------+-----------------------------------------------------------+
        | ['科', '技', '全', '方', '位', '资', '讯', '智', '能',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['对', '，', '输', '给', '一', '个', '女', '人', '，',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM... |
        | ['今', '天', '下', '午', '起', '来', '看', '到', '外',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['今', '年', '拜', '年', '不', '短', '信', '，', '就',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['浑', '身', '酸', '疼', '，', '两', '腿', '无', '力',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['明', '显', '紧', '张', '状', '态', '没', '出', '来',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['三', '十', '年', '前', '，', '老', '爹', '带', '我',...  | ['O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM', 'O... |
        | ['好', '活', '动', '呀', '，', '给', '力', '的', '商',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ['人', '生', '如', '戏', '，', '导', '演', '是', '自',...  | ['O', 'O', 'O', 'O', 'O', 'B-PER.NOM', 'I-PER.NOM', 'O... |
        | ['听', '说', '小', '米', '开', '卖', '了', '，', '刚',...  | ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'... |
        | ...                                                       | ...                                                       |
        +-----------------------------------------------------------+-----------------------------------------------------------+
    """

    for k,v in datasets.items():
        print('{}:{}'.format(k,len(v)))
    # print(*list(datasets.keys()))
    vocabs = {}
    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    cnt = 0 # datasets 就3个键值对，分别是 train:[] , dev:[], test:[], 
    for item in datasets.items():
        if cnt < 100:
            print(item)
            cnt += 1
        else: 
            break

    for k,v in datasets.items():  # 处理键值对
        # ignore the word segmentation tag
        v.apply_field(lambda x: [w[0] for w in x],'chars','chars')
        v.apply_field(get_bigrams,'chars','bigrams')  # 感觉这里的效果就是将连续的两个字拼在一起，也就是所谓的 bigrams 

    
    char_vocab.from_dataset(datasets['train'],field_name='chars',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    label_vocab.from_dataset(datasets['train'],field_name='target')
    print('label_vocab:{}\n{}'.format(len(label_vocab),label_vocab.idx2word))


    for k,v in datasets.items():
        # v.set_pad_val('target',-100)
        v.add_seq_len('chars',new_field_name='seq_len')


    vocabs['char'] = char_vocab
    vocabs['label'] = label_vocab


    bigram_vocab.from_dataset(datasets['train'],field_name='bigrams',no_create_entry_dataset=[datasets['dev'],datasets['test']])
    if index_token:
        char_vocab.index_dataset(*list(datasets.values()), field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(*list(datasets.values()),field_name='bigrams',new_field_name='bigrams')
        label_vocab.index_dataset(*list(datasets.values()), field_name='target', new_field_name='target')

    # for k,v in datasets.items():
    #     v.set_input('chars','bigrams','seq_len','target')
    #     v.set_target('target','seq_len')

    vocabs['bigram'] = bigram_vocab

    embeddings = {}

    if unigram_embedding_path is not None:
        unigram_embedding = StaticEmbedding(char_vocab, model_dir_or_name=unigram_embedding_path,
                                            word_dropout=char_word_dropout,
                                            min_freq=char_min_freq,only_train_min_freq=only_train_min_freq,)
        embeddings['char'] = unigram_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, model_dir_or_name=bigram_embedding_path,
                                           word_dropout=0.01,
                                           min_freq=bigram_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings




if __name__ == '__main__':
    pass