import datetime as dt
from fastNLP.core.tester import Tester
import fitlog
use_fitlog = True
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed,'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)

import sys
sys.path.append('../')
from load_data import *
import argparse
from paths import *
from fastNLP.core import Trainer
from fastNLP.core import Callback
from V1.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import MetricBase, SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback
from fastNLP import FitlogCallback
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
import fitlog
from fastNLP import logger
from utils import get_peking_time
from V1.add_lattice import equip_chinese_ner_with_lexicon
from load_data import load_toy_ner

import traceback
import warnings
from utils import print_info
from fastNLP_module import BertEmbedding
from V1.models import BERT_SeqLabel


parser = argparse.ArgumentParser()
# -----------------------------------------如下一大部分全都是修改参数相关------------------------------------
# -----------------------------------------performance inrelevant-----------------------------------------
parser.add_argument('--update_every',type=int,default=1)
parser.add_argument('--status',choices=['train','test'],default='train')
parser.add_argument('--use_bert',type=int,default=1)
parser.add_argument('--only_bert',type=int,default=0)
parser.add_argument('--fix_bert_epoch',type=int,default=20)
parser.add_argument('--after_bert',default='mlp',choices=['lstm','mlp'])
parser.add_argument('--msg',default='11266')
parser.add_argument('--train_clip',default=False,help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0,type=int)
parser.add_argument('--gpumm',default=False,help='查看显存')
parser.add_argument('--see_convergence',default=False)
parser.add_argument('--see_param',default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=1080956,type=int)
parser.add_argument('--test_train',default=False)
parser.add_argument('--number_normalized',type=int,default=0,
                    choices=[0,1,2,3],help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name',default='yj',choices=['lk','yj'])
parser.add_argument('--use_pytorch_dropout',type=int,default=0)
parser.add_argument('--char_min_freq',default=1,type=int)
parser.add_argument('--bigram_min_freq',default=1,type=int)
parser.add_argument('--lattice_min_freq',default=1,type=int)
parser.add_argument('--only_train_min_freq',default=True)
parser.add_argument('--only_lexicon_in_train',default=False)
parser.add_argument('--word_min_freq',default=1,type=int)


# hyper of training
parser.add_argument('--early_stop',default=25,type=int)
parser.add_argument('--epoch', default=200, type=int) 
parser.add_argument('--batch', default=5, type=int) # 我修改了默认batch的大小
parser.add_argument('--optim', default='sgd', help='sgd|adam')
#parser.add_argument('--lr', default=6e-4, type=float)
parser.add_argument('--lr', default=6e-3, type=float)
parser.add_argument('--bert_lr_rate',default=0.05,type=float)
parser.add_argument('--embed_lr_rate',default=1,type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init',default='uniform',help='norm|uniform')
parser.add_argument('--self_supervised',default=False)
parser.add_argument('--weight_decay',default=0,type=float)
parser.add_argument('--norm_embed',default=True)
parser.add_argument('--norm_lattice_embed',default=True)
parser.add_argument('--warmup',default=0.1,type=float)

# hyper of model
parser.add_argument('--model',default='transformer',help='lstm|transformer')
parser.add_argument('--lattice',default=1,type=int)
parser.add_argument('--use_bigram', default=1,type=int)
parser.add_argument('--hidden', default=-1,type=int)
parser.add_argument('--ff', default=3,type=int)
parser.add_argument('--layer', default=1,type=int)
parser.add_argument('--head', default=8,type=int)
parser.add_argument('--head_dim',default=20,type=int)
parser.add_argument('--scaled',default=False)
parser.add_argument('--ff_activate',default='relu',help='leaky|relu')

parser.add_argument('--k_proj',default=False)
parser.add_argument('--q_proj',default=True)
parser.add_argument('--v_proj',default=True)
parser.add_argument('--r_proj',default=True)

parser.add_argument('--attn_ff',default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos',default=False)
parser.add_argument('--use_rel_pos',default=True)
#相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared',default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm',default=False)
parser.add_argument('--rel_pos_init',default=1)
parser.add_argument('--four_pos_shared',default=True,help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion',default='ff_two',choices=['ff','attn','gate','ff_two','ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared',default=True,help='是不是要共享4个位置融合之后形成的pos')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout =  -1
parser.add_argument('--embed_dropout_before_pos',default=False)
parser.add_argument('--embed_dropout', default=0.5,type=float)
parser.add_argument('--gaz_dropout',default=0.5,type=float)
parser.add_argument('--output_dropout', default=0.3,type=float)
parser.add_argument('--pre_dropout', default=0.5,type=float)
parser.add_argument('--post_dropout', default=0.3,type=float)
parser.add_argument('--ff_dropout', default=0.15,type=float)
parser.add_argument('--ff_dropout_2', default=-1,type=float,help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout',default=0,type=float)
parser.add_argument('--embed_dropout_pos',default='0')
parser.add_argument('--abs_pos_fusion_func',default='nonlinear_add',
                    choices=['add','concat','nonlinear_concat','nonlinear_add','concat_nonlinear','add_nonlinear'])

parser.add_argument('--dataset', default='ontonotes', help='weibo|resume|ontonotes|msra')
# parser.add_argument('--debug',default=1)

# -----------------------------------------参数设置部分结束-----------------------------------------
# -------------------------------------------------------------------------------------------------
args = parser.parse_args()
if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout>0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout



if args.lattice and args.use_rel_pos:
    args.train_clip = True

# fitlog.commit(__file__,fit_msg='绝对位置用新的了')
fitlog.set_log_dir('logs')
now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time),level='info')
if args.test_batch == -1:
    args.test_batch = args.batch//2
fitlog.add_hyper(now_time,'time')
if args.debug:
    # args.dataset = 'toy'
    pass


if args.device!='cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = False  # 用以表示加载数据的时候是否重新加载


for k,v in args.__dict__.items():
    print_info('{}:{}'.format(k,v))

raw_dataset_cache_name = os.path.join('cache',args.dataset
                                      +'_trainClip:{}'.format(args.train_clip)
                                      +'bgminfreq_{}'.format(args.bigram_min_freq)
                                      +'char_min_freq_{}'.format(args.char_min_freq)
                                      +'word_min_freq_{}'.format(args.word_min_freq)
                                      +'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      +'number_norm{}'.format(args.number_normalized)
                                      + 'load_dataset_seed{}'.format(load_dataset_seed))


if args.dataset == 'ontonotes':
    datasets,vocabs,embeddings = load_ontonotes4ner(ontonote4ner_cn_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,train_clip=args.train_clip,
                                                    _cache_fp=raw_dataset_cache_name,
                                                    char_min_freq=args.char_min_freq,
                                                    bigram_min_freq=args.bigram_min_freq,
                                                    only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'resume':
    datasets,vocabs,embeddings = load_resume_ner(resume_ner_path,yangjie_rich_pretrain_unigram_path,yangjie_rich_pretrain_bigram_path,
                                                    _refresh=refresh_data,index_token=False,
                                                 _cache_fp=raw_dataset_cache_name,
                                                 char_min_freq=args.char_min_freq,
                                                 bigram_min_freq=args.bigram_min_freq,
                                                 only_train_min_freq=args.only_train_min_freq
                                                    )
elif args.dataset == 'weibo':
    datasets,vocabs,embeddings = load_weibo_ner(weibo_ner_path,
                                                yangjie_rich_pretrain_unigram_path,
                                                yangjie_rich_pretrain_bigram_path,
                                                _refresh=refresh_data,
                                                _cache_fp=raw_dataset_cache_name,
                                                index_token=False,                                                
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq
                                                )

# 添加天池的数据集
# embeddings 的作用是什么？
elif args.dataset == 'tianchi':
        datasets,vocabs,embeddings = load_tianchi_ner(tianchi_ner_path, # 文本数据集
                                                #yangjie_rich_pretrain_unigram_path, # 这是两个预训练好的结果
                                                #yangjie_rich_pretrain_bigram_path,
                                                _refresh=refresh_data, # 下面两行是关于cache 的参数
                                                _cache_fp=raw_dataset_cache_name,
                                                index_token=False,               
                                                char_min_freq=args.char_min_freq,
                                                bigram_min_freq=args.bigram_min_freq,
                                                only_train_min_freq=args.only_train_min_freq
                                                )                                                    

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff


if args.dataset == 'ontonotes':
    args.update_every = 2
    pass


if args.lexicon_name == 'lk':
    yangjie_rich_pretrain_word_path = lk_word_path_2

#print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))
print('用的词表的路径:{}'.format(yangjie_rich_pretrain_word_path))

# 需要根据天池的任务进行修改，我觉得不应该读取 yangjie_rich_pretrian_word_path
# 而应该读取一个和任务相关的词典数据
w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name)
                                              )

cache_name = os.path.join('cache',(args.dataset+'_lattice'+'_only_train:{}'+
                          '_trainClip:{}'+'_norm_num:{}'
                                   +'char_min_freq{}'+'bigram_min_freq{}'+'word_min_freq{}'+'only_train_min_freq{}'
                                   +'number_norm{}'+'lexicon_{}'+'load_dataset_seed_{}')
                          .format(args.only_lexicon_in_train,
                          args.train_clip,args.number_normalized,args.char_min_freq,
                                  args.bigram_min_freq,args.word_min_freq,args.only_train_min_freq,
                                  args.number_normalized,args.lexicon_name,load_dataset_seed))

'''
vocabs['lattice'] 这个就是在 equip_chinese_ner_with_lexicon() 方法中生成的，其中的内容就是词语
'''                                  
datasets,vocabs,embeddings = equip_chinese_ner_with_lexicon(datasets,
                                                            vocabs,
                                                            embeddings,  # 传入刚才读取的embedding
                                                            w_list,   # 这个也需要根据tianchi任务进行修改
                                                            word_embedding_path = None, 
                                                            # yangjie_rich_pretrain_word_path,  # 用的是这个词表的embedding
                                                            _refresh=refresh_data,
                                                            _cache_fp=cache_name,
                                                            only_lexicon_in_train=args.only_lexicon_in_train,
                                                            #原论文中是使用的下面这个路径的词典，但是针对天池数据，
                                                            #我将其设置 cn-sgns-literature-word ，也是一个与训练的包，可能会含有自己
                                                            #需要的关键词多一些
                                                            #word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                            #word_char_mix_embedding_path=tianchi_pretrain_words_path,
                                                            word_char_mix_embedding_path = 'cn-sgns-literature-word',
                                                            number_normalized=args.number_normalized,
                                                            lattice_min_freq=args.lattice_min_freq,
                                                            only_train_min_freq=args.only_train_min_freq)
print(vocabs['lattice'])

print('train:{}'.format(len(datasets['train'])))
avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
for k,v in datasets.items():
    max_seq_len = 0
    max_lex_num = 0
    max_seq_lex = 0
    max_seq_len_i = -1
    for i in range(len(v)):
        if max_seq_len < v[i]['seq_len']:
            max_seq_len = v[i]['seq_len']
            max_seq_len_i = i
        # max_seq_len = max(max_seq_len,v[i]['seq_len'])
        max_lex_num = max(max_lex_num,v[i]['lex_num'])
        max_seq_lex = max(max_seq_lex,v[i]['lex_num']+v[i]['seq_len'])

        avg_seq_len+=v[i]['seq_len']
        avg_lex_num+=v[i]['lex_num']
        avg_seq_lex+=(v[i]['seq_len']+v[i]['lex_num'])
        if k == 'train':
            train_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            train_seq.append(v[i]['seq_len'])
            if v[i]['seq_len'] >200: # 
                print('train里这个句子char长度已经超了200了')
                print(''.join(list(map(lambda x:vocabs['char'].to_word(x),v[i]['chars']))))
            else:
                if v[i]['seq_len']+v[i]['lex_num']>400:
                    print('train里这个句子char长度没超200，但是总长度超了400')
                    print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
        if k == 'dev':
            dev_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            dev_seq.append(v[i]['seq_len'])
        if k == 'test':
            test_seq_lex.append(v[i]['lex_num']+v[i]['seq_len'])
            test_seq.append(v[i]['seq_len'])

# 输出最长的句子是为了什么？是单个句子单个句子的训练吗？
    print('{} 最长的句子是:{}'.format(k,list(map(lambda x:vocabs['char'].to_word(x),v[max_seq_len_i]['chars']))))
    print('{} max_seq_len:{}'.format(k,max_seq_len))
    print('{} max_lex_num:{}'.format(k, max_lex_num))
    print('{} max_seq_lex:{}'.format(k, max_seq_lex))


# max_seq_len = max(max(datasets['train']['seq_len']),max(datasets['dev']['seq_len']),max(datasets['test']['seq_len']))
import copy
max_seq_len = max(* map(lambda x:max(x['seq_len']),datasets.values()))

show_index = 4
# 程序在下面的代码中出现问题
print('raw_chars:{}'.format(list(datasets['train'][show_index]['raw_chars'])))
print('lexicons:{}'.format(list(datasets['train'][show_index]['lexicons'])))
print('lattice:{}'.format(list(datasets['train'][show_index]['lattice'])))

for i in range(0,100):
    print('raw_lattice:{}'.format(list(map(lambda x:vocabs['lattice'].to_word(x),
                                  list(datasets['train'][i]['lattice'])))))
print('lex_s:{}'.format(list(datasets['train'][show_index]['lex_s'])))
print('lex_e:{}'.format(list(datasets['train'][show_index]['lex_e'])))
print('pos_s:{}'.format(list(datasets['train'][show_index]['pos_s'])))
print('pos_e:{}'.format(list(datasets['train'][show_index]['pos_e'])))



for k, v in datasets.items():    
    if args.lattice:
        v.set_input('lattice','bigrams','seq_len','target') # 将filed_name的field 设置为input
        v.set_input('lex_num','pos_s','pos_e')
        v.set_target('target','seq_len')   # 将field_names的field设置为target
        v.set_pad_val('lattice',vocabs['lattice'].padding_idx)
    else:
        v.set_input('chars','bigrams','seq_len','target')
        v.set_target('target', 'seq_len')

# ============================================下面这部分像是对应模型的 Add & Norm 部分，但是没有体现Add啊 ？
from utils import norm_static_embedding
# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed>0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)

if args.norm_lattice_embed>0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k,v in embeddings.items():
        norm_static_embedding(v,args.norm_embed)


mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
if args.debug or args.gpumm:
    fitlog.debug()
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

torch.backends.cudnn.benchmark = False
fitlog.set_rng_seed(args.seed)
torch.backends.cudnn.benchmark = False


fitlog.add_hyper(args)
    
'''
01.bert_embedding 有什么作用？
02.和embedding 的区别是什么？
'''
if args.model == 'transformer':
    if args.lattice:
        if args.use_bert:
            # BertEmbedding in fastNLP
            # 使用的embedding 模型的简称是 cn-wwm。这个信息可以在fastNLP 的官网中找到相关信息
            # 但是很好奇，为什么这里是用 vocabs['lattice'] ? 而不是 vocabs['char']  ?
            bert_embedding = BertEmbedding(vocabs['lattice'],model_dir_or_name='cn-wwm',requires_grad=False,
                                           word_dropout=0.01)
        else:
            bert_embedding = None

        if args.only_bert: # 仅仅使用bert模型
            model = BERT_SeqLabel(bert_embedding,len(vocabs['label']),vocabs,args.after_bert)
        else: # 用的是bert+flat这个模型
            # 在这里用了 embeddings 的信息，用它去构建一个实例
            # 为什么选择 embeddings['lattice'] 和 embeddings['bigram'] 作为参数？
            model = Lattice_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                         args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                         args.learn_pos, args.add_pos,
                                         args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                         mode,device,vocabs,
                                         max_seq_len=max_seq_len,
                                         rel_pos_shared=args.rel_pos_shared,
                                         k_proj=args.k_proj,
                                         q_proj=args.q_proj,
                                         v_proj=args.v_proj,
                                         r_proj=args.r_proj,
                                         self_supervised=args.self_supervised,
                                         attn_ff=args.attn_ff,
                                         pos_norm=args.pos_norm,
                                         ff_activate=args.ff_activate,
                                         abs_pos_fusion_func=args.abs_pos_fusion_func,
                                         embed_dropout_pos=args.embed_dropout_pos,
                                         four_pos_shared=args.four_pos_shared,
                                         four_pos_fusion=args.four_pos_fusion,
                                         four_pos_fusion_shared=args.four_pos_fusion_shared,
                                         bert_embedding=bert_embedding
                                        )
    else: # 不使用lattice 
        model = Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden, len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos,args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled,dropout,args.use_bigram,
                                     mode,device,vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos
                                     )
    
elif args.model =='lstm':
    model = LSTM_SeqLabel_True(embeddings['char'],embeddings['bigram'],embeddings['bigram'],args.hidden,
                               len(vocabs['label']),
                          bidirectional=True,device=device,
                          embed_dropout=args.embed_dropout,output_dropout=args.output_dropout,use_bigram=True,
                          debug=args.debug)

for n,p in model.named_parameters():
    print('{}:{}'.format(n,p.size()))


# print('这次让pytorch默认初始化transformer')
with torch.no_grad():
    print_info('{}init pram{}'.format('*'*15,'*'*15))
    for n,p in model.named_parameters():
        if 'bert' not in n and 'embedding' not in n and 'pos' not in n and 'pe' not in n \
                and 'bias' not in n and 'crf' not in n and p.dim()>1:
            try:
                if args.init == 'uniform':
                    nn.init.xavier_uniform_(p)
                    print_info('xavier uniform init:{}'.format(n))
                elif args.init == 'norm':
                    print_info('xavier norm init:{}'.format(n))
                    nn.init.xavier_normal_(p)
            except:
                print_info(n)
                exit(1208)
    print_info('{}init pram{}'.format('*' * 15, '*' * 15))

loss = LossInForward()
encoding_type = 'bmeso'
if args.dataset == 'weibo':
    encoding_type = 'bio'
if args.dataset == 'tianchi': # 根据不同的数据集，定义不同的标注方式
    encoding_type = 'bio'
f1_metric = SpanFPreRecMetric(vocabs['label'],pred='pred',target='target',seq_len='seq_len',encoding_type=encoding_type)
acc_metric = AccuracyMetric(pred='pred',target='target',seq_len='seq_len',)
acc_metric.set_metric_name('label_acc')
metrics = [
    f1_metric,
    acc_metric
]
if args.self_supervised:
    chars_acc_metric = AccuracyMetric(pred='chars_pred',target='chars_target',seq_len='seq_len')
    chars_acc_metric.set_metric_name('chars_acc')
    metrics.append(chars_acc_metric)

if args.see_param:
    for n,p in model.named_parameters():
        print_info('{}:{}'.format(n,p.size()))
    print_info('see_param mode: finish')
    if not args.debug:
        exit(1208)
datasets['train'].apply
if args.see_convergence:
    print_info('see_convergence = True')
    print_info('so just test train acc|f1')
    datasets['train'] = datasets['train'][:100]
    if args.optim == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    trainer = Trainer(datasets['train'], model, optimizer, loss, args.batch,
                      n_epochs=args.epoch, dev_data=datasets['train'], metrics=metrics,
                      device=device, dev_batch_size=args.test_batch)

    trainer.train()
    exit(1208)


if not args.only_bert:
    if not args.use_bert:
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(lambda x:id(x) not in embedding_param_ids,model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]
    else: 
        bert_embedding_param = list(model.bert_embedding.parameters())
        bert_embedding_param_ids = list(map(id,bert_embedding_param))
        bigram_embedding_param = list(model.bigram_embed.parameters())
        gaz_embedding_param = list(model.lattice_embed.parameters())
        embedding_param = bigram_embedding_param
        if args.lattice:
            gaz_embedding_param = list(model.lattice_embed.parameters())
            embedding_param = embedding_param+gaz_embedding_param
        embedding_param_ids = list(map(id,embedding_param))
        non_embedding_param = list(filter(
            lambda x:id(x) not in embedding_param_ids and id(x) not in bert_embedding_param_ids,
                                          model.parameters()))
        param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate},
                  {'params':bert_embedding_param,'lr':args.bert_lr_rate*args.lr}]
else:
    non_embedding_param = model.parameters()
    embedding_param = []
    param_ = [{'params': non_embedding_param}, {'params': embedding_param, 'lr': args.lr * args.embed_lr_rate}]




if args.optim == 'adam':
    optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
elif args.optim == 'sgd':
    # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD(param_,lr=args.lr,momentum=args.momentum,
                          weight_decay=args.weight_decay)

if args.dataset == 'msra':
    datasets['dev']  = datasets['test']
fitlog_evaluate_dataset = {'test':datasets['test']}
if args.test_train:
    fitlog_evaluate_dataset['train'] = datasets['train']
evaluate_callback = FitlogCallback(fitlog_evaluate_dataset,verbose=1)
lrschedule_callback = LRScheduler(lr_scheduler=LambdaLR(optimizer, lambda ep: 1 / (1 + 0.05*ep) ))
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)

class Unfreeze_Callback(Callback):
    def __init__(self,bert_embedding,fix_epoch_num):
        super().__init__()
        self.bert_embedding = bert_embedding
        self.fix_epoch_num = fix_epoch_num
        assert self.bert_embedding.requires_grad == False

    def on_epoch_begin(self):
        if self.epoch == self.fix_epoch_num+1:
            self.bert_embedding.requires_grad = True


callbacks = [
        evaluate_callback,   # 好奇evaluate_callback的值是
        lrschedule_callback,
        clip_callback
    ]
if args.use_bert:
    if args.fix_bert_epoch != 0:
        callbacks.append(Unfreeze_Callback(bert_embedding,args.fix_bert_epoch))
    else:
        bert_embedding.requires_grad = True
callbacks.append(EarlyStopCallback(args.early_stop))
if args.warmup > 0 and args.model == 'transformer':
    callbacks.append(WarmupCallback(warmup=args.warmup))


class record_best_test_callback(Callback):
    def __init__(self,trainer,result_dict):
        super().__init__()
        self.trainer222 = trainer
        self.result_dict = result_dict

    def on_valid_end(self, eval_result, metric_key, optimizer, better_result):
        print(eval_result['data_test']['SpanFPreRecMetric']['f'])

print(torch.rand(size=[3,3],device=device))


if args.status == 'train':        
    # 如果觉得dataset['train'] 较大，可以执行如下操作删除其 instance
    # for i in range(len(datasets['train'])-1,2,-1):
    #     datasets['train'].delete_instance(i)
    #print(datasets['train'])
    '''
    上面这个数据集的格式，我将其写入到了一个文件中，文件地址是： V1/train_exam.txt
    '''
    trainer = Trainer(datasets['train'],
                    model,
                    optimizer,
                    loss,
                    args.batch,
                    n_epochs=args.epoch,
                      dev_data=datasets['dev'],
                      metrics=metrics,
                      device=device,
                      callbacks=callbacks,
                      dev_batch_size=args.test_batch,
                      test_use_tqdm=False,
                      check_code_level=-1,
                      update_every=args.update_every,
                      save_path="./model") # 保存模型
    trainer.train()


if args.status == 'test': # 如果是做测试 
    # 因为这里使用的是fastNLP ，故加载模型的方法有些区别。直接load就可以了   
    model = torch.load("/home/liushen/program/Flat-Lattice-Transformer/V1/model/best_Lattice_Transformer_SeqLabel_f_2020-10-09-17-59-43") # 加载训练好的模型
    tester = Tester(datasets['test'],model,
                      metrics=metrics,
                      device=device)
    res = tester.test()
    print(res)