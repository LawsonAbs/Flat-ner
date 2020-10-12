'''
Modify: LawsonAbs
Date: 2020-10-06 20:25:50
LastEditTime: 2020-10-12 16:35:45
FilePath: /Flat-Lattice-Transformer/paths.py
'''

# 需要修改下面的路径
"""
1.yangjie_rich_pretrain_unigram_path 和 1.yangjie_rich_pretrain_bigram_path 所代表的文件中都是有繁体字的
unigram_path 是单个字的embedding， bigram 是两个字的embedding 
"""
yangjie_rich_pretrain_unigram_path = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/gigaword_chn.all.a2b.uni.ite50.vec'
yangjie_rich_pretrain_bigram_path = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/gigaword_chn.all.a2b.bi.ite50.vec'
yangjie_rich_pretrain_word_path = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/ctb.50d.vec'

# 下面这个文件是随着程序运行自动生成出来的
yangjie_rich_pretrain_char_and_word_path = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/yangjie_word_char_mix.txt'

# 我针对自己的任务输出搞了一个embedding，当然这个embedding并非是我训练出来的，而是我“随机生成”的（我从yangjie_word_char_mix.txt 中把汉语词语做了替换）# 就得到了“适用于”本任务的embedding
tianchi_pretrain_words_path = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/embedding.txt'
lk_word_path_2 = '/home/lawson/program/Flat-Lattice-Transformer/data/pretrain/sgns.merge.word_2'


ontonote4ner_cn_path = '/home/lawson/program/Flat-Lattice-Transformer/data/corpus/OntoNote4NER'
msra_ner_cn_path = '/home/lawson/program/Flat-Lattice-Transformer/data/corpus/MSRANER'
resume_ner_path = '/home/lawson/program/Flat-Lattice-Transformer/data/corpus/ResumeNER'
weibo_ner_path = '/home/lawson/program/Flat-Lattice-Transformer/data/corpus/WeiboNER'
tianchi_ner_path = '/home/lawson/program/Flat-Lattice-Transformer/data/corpus/TianChiNER'