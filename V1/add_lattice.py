from fastNLP import cache_results

'''
description:  我对word_embedding_path  和word_char_mix_embedding_path 二者之间的区别不是很懂
param {type} 
01.word_embedding_path 
02.word_char_mix_embedding_path 
return {type} 
'''
@cache_results(_cache_fp='need_to_defined_fp',_refresh=True)
def equip_chinese_ner_with_lexicon(datasets,
                                    vocabs,
                                    embeddings,
                                    w_list,
                                    word_embedding_path=None,  
                                   only_lexicon_in_train=False,
                                   word_char_mix_embedding_path=None, # 字和词的embedding信息
                                   number_normalized=False,
                                   lattice_min_freq=1,
                                   only_train_min_freq=0):
    from fastNLP.core import Vocabulary
    def normalize_char(inp):
        result = []
        for c in inp:
            if c.isdigit():
                result.append('0')
            else:
                result.append(c)

        return result

    def normalize_bigram(inp):
        result = []
        for bi in inp:
            tmp = bi
            if tmp[0].isdigit():
                tmp = '0'+tmp[:1]
            if tmp[1].isdigit():
                tmp = tmp[0]+'0'

            result.append(tmp)
        return result

    if number_normalized == 3:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    if only_lexicon_in_train:
        print('已支持只加载在trian中出现过的词汇')

    def get_skip_path(chars, w_trie):
        sentence = ''.join(chars)
        result = w_trie.get_lexicon(sentence)
        # print(result)

        return result
    from V0.utils_ import Trie
    from functools import partial
    from fastNLP.core import Vocabulary    
    from fastNLP_module import StaticEmbedding
    from fastNLP import DataSet
    a = DataSet()
    w_trie = Trie()
    for w in w_list:
        w_trie.insert(w)


    if only_lexicon_in_train:
        lexicon_in_train = set()
        for s in datasets['train']['chars']:
            lexicon_in_s = w_trie.get_lexicon(s)
            for s,e,lexicon in lexicon_in_s:
                lexicon_in_train.add(''.join(lexicon))

        print('lexicon in train:{}'.format(len(lexicon_in_train)))
        print('i.e.: {}'.format(list(lexicon_in_train)[:10]))
        w_trie = Trie()
        for w in lexicon_in_train:
            w_trie.insert(w)

    import copy
    for k,v in datasets.items():
        v.apply_field(partial(get_skip_path,w_trie=w_trie),'chars','lexicons')
        v.apply_field(copy.copy, 'chars','raw_chars')
        v.add_seq_len('lexicons','lex_num')
        v.apply_field(lambda x:list(map(lambda y: y[0], x)), 'lexicons', 'lex_s')
        v.apply_field(lambda x: list(map(lambda y: y[1], x)), 'lexicons', 'lex_e')


    if number_normalized == 1:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

    if number_normalized == 2:
        for k,v in datasets.items():
            v.apply_field(normalize_char,'chars','chars')
        vocabs['char'] = Vocabulary()
        vocabs['char'].from_dataset(datasets['train'], field_name='chars',
                                no_create_entry_dataset=[datasets['dev'], datasets['test']])

        for k,v in datasets.items():
            v.apply_field(normalize_bigram,'bigrams','bigrams')
        vocabs['bigram'] = Vocabulary()
        vocabs['bigram'].from_dataset(datasets['train'], field_name='bigrams',
                                  no_create_entry_dataset=[datasets['dev'], datasets['test']])


    def concat(ins):
        chars = ins['chars']
        lexicons = ins['lexicons']
        result = chars + list(map(lambda x:x[2],lexicons))
        # print('lexicons:{}'.format(lexicons))
        # print('lex_only:{}'.format(list(filter(lambda x:x[2],lexicons))))
        # print('result:{}'.format(result))
        return result

    def get_pos_s(ins):
        lex_s = ins['lex_s']
        seq_len = ins['seq_len']
        pos_s = list(range(seq_len)) + lex_s

        return pos_s

    def get_pos_e(ins):
        lex_e = ins['lex_e']
        seq_len = ins['seq_len']
        pos_e = list(range(seq_len)) + lex_e

        return pos_e



    for k,v in datasets.items():
        v.apply(concat,new_field_name='lattice')
        v.set_input('lattice')
        v.apply(get_pos_s,new_field_name='pos_s')
        v.apply(get_pos_e, new_field_name='pos_e')
        v.set_input('pos_s','pos_e')


    word_vocab = Vocabulary()
    word_vocab.add_word_lst(w_list)
    vocabs['word'] = word_vocab

    lattice_vocab = Vocabulary()
    lattice_vocab.from_dataset(datasets['train'],field_name='lattice',
                               no_create_entry_dataset=[v for k,v in datasets.items() if k != 'train'])
    vocabs['lattice'] = lattice_vocab

    """
    1.word_embedding_path 这个参数到底是用做什么的？
    我将其设置成了 None。但是如果为None，那么embedding['word']没有了还可以吗？
    
    2.StaticEmbedding：
    给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据(只会将出现在vocab中的词抽取出来， 如果没有找到，则会随机初始化一个值(但如果该word是被标记为no_create_entry的话，则不会单独创建一个值，而是会被指向unk的index))
    """
    if word_embedding_path is not None:
        word_embedding = StaticEmbedding(word_vocab,word_embedding_path,word_dropout=0)
        embeddings['word'] = word_embedding

    if word_char_mix_embedding_path is not None:
        lattice_embedding = StaticEmbedding(lattice_vocab, word_char_mix_embedding_path,word_dropout=0.01,
                                            min_freq=lattice_min_freq,only_train_min_freq=only_train_min_freq)
        embeddings['lattice'] = lattice_embedding

    vocabs['char'].index_dataset(* (datasets.values()),
                             field_name='chars', new_field_name='chars')
    vocabs['bigram'].index_dataset(* (datasets.values()),
                               field_name='bigrams', new_field_name='bigrams')
    vocabs['label'].index_dataset(* (datasets.values()),
                              field_name='target', new_field_name='target')
    vocabs['lattice'].index_dataset(* (datasets.values()),
                                    field_name='lattice', new_field_name='lattice')

    return datasets,vocabs,embeddings