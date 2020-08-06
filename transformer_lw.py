# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    
)
#TransformerDecoderLayer_lw
#TransformerEncoderLayer_lw
#import .transformer_layer_lw
from .transformer_layer_lw import TransformerEncoderLayer_lw
from .transformer_layer_lw import TransformerDecoderLayer_lw
from .transformer_layer_lw import TransformerEncoderLayer_lw_mod2
from .transformer_layer_lw import TransformerDecoderLayer_lw_mod2

from torch import Tensor


from fairseq.modules.transformer_sentence_encoder import init_bert_params

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

import numpy as np
#pvtzero = torch.from_numpy(np.array(0.00001)).cuda()
pvtzero=0.00001


def angular_distance(vec1,vec2):
    vec1=vec1.cuda()
    vec2=vec2.cuda()
    Pi=3.1415926
    cos_sim=F.cosine_similarity(vec1, vec2,dim=-1)

    #cos_sim=F.relu(cos_sim) #avoid numerical overflow of -0

    cos_sim=cos_sim*0.99 #avoid numerical overflow of 1
    dist=torch.acos(cos_sim)/Pi
    return dist.cuda()



def mask_mean(tmp_x,padding_mask, dim, src_lengths=None):
    #feature dimension is the last dimension
    #dim is the seqlen dimension, 1 for oringal bsz x seqlen x fea,  0 for seqlen x bsz x fea 
    #if input is seqlen x bsz x fea, first transpose padding_mask to seqlen x bsz
    #then .sum(dim=0) src_lengths_tmp = bsz
    tmp_x=tmp_x.cuda()

    if src_lengths==None:
        src_lengths=(padding_mask<=0).sum(dim=dim)

    #print ('\n\n\n####################mask_mean################## \n src_lengths = {}\n\n\n'.format(src_lengths))    
    mean_tmp3=tmp_x.masked_fill_(padding_mask.unsqueeze(-1), 0.0).sum(dim = dim)/((src_lengths +pvtzero).unsqueeze(-1))
    return mean_tmp3.cuda()

class sentemb_layer_lw(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        output_dim,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, output_dim)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

#args.activation_fn = getattr(args, 'activation_fn', 'gelu')
#args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
#args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)




class key_master(nn.Module):
    """key master."""
    def __init__(
        self,
        args,
        input_dim,
        key_dim,
        output_dim,
        eos,
        activation_fn="tanh",
        pooler_dropout=0.0,
    ):
        super().__init__()
        
        print ('\n################\nliwei comment keylayer =4 fixkey, 4 dymkey from layer6 and 4 dymkey from layer4, the 8 key is concatenate with 12 keylayer output to from 20 outputs and then use master key to sum them\n############### into one vector\n')
        #torch.Tensor([max_lengths] * batch_size).to(torch.int32).to(device)
        #self.symbol_embeddings = nn.Parameter(torch.FloatTensor(2, word_embed_dim))
        #self.keys = torch.Tensor([max_lengths] * batch_size).to(torch.int32).to(device)
        self.fix_keys_num=4 #mean, max, first, last in 2 layers
         
        self.key_dim=key_dim
        self.master_key=nn.Parameter(torch.FloatTensor(1,self.key_dim))
        self.fix_keys =nn.Parameter(torch.FloatTensor(self.fix_keys_num,self.key_dim))
        #self.dym_keys =nn.Parameter(torch.FloatTensor(dym_keys_num,key_dim))
        self.dym_keys=None

        #TransformerDecoderLayer_lw
        self.keys_layer=self.build_decoder_layer(args, no_encoder_attn=False)
        self.master_layer=self.build_encoder_layer(args)
        
        
        
        print ('\n############\nself.fix_keys_num = {} self.fix_keys.size = {}  args.sentemb_mod ={}\n##########\n'\
        .format(self.fix_keys_num, self.fix_keys.size,args.sentemb_mod))
        
        self.master_key.data.normal_(mean=0.0, std=0.02)
        self.fix_keys.data.normal_(mean=0.0, std=0.02)
        #self.encoder.dictionary.eos()
        self.eos=eos
        
        if args.init_mod=='BERT':
            self.apply(init_bert_params)
        
    def combine_keys(self,dym_keys,bsz):
        #dym_keys t x bsz x fea
        #fix_keys keys_num x bsz x fea
        #input = torch.cat((x, encoder_sentemb_tmp1, l), dim=-1)
        
        
        fix_keys=(self.fix_keys.view(self.fix_keys_num,1,self.key_dim)).expand(-1,bsz,-1)
        #assert (fix_keys.size()=(self.fix_keys_num,self.key_dim))
        #dym_keys=torch.cat(dym_keys, dim=0)
        #print ('dym_keys.size() {} fix_keys.size {}'.format(dym_keys.size(),fix_keys.size()))
        keys=torch.cat((fix_keys, dym_keys), dim=0)
        return keys
        
    @classmethod
    def get_dym_keys(cls,encoder_out,encoder_padding_mask,src_tokens,eos, fp16,**kwargs):
        #=========================================================================================================
        #encoder_out=encoder_out.half()
        seqlen,bsz,fea = encoder_out.size() #liwei mod

        #sentemb=mask_mean(tmp_x=x,padding_mask=encoder_padding_mask.transpose(0,1),dim=0,src_lengths=src_lengths)
        mean_key=mask_mean(tmp_x=encoder_out,padding_mask=encoder_padding_mask.transpose(0,1), dim=0, src_lengths=None)
        #mean_key=encoder_out.mean()
        max_key=encoder_out.max(dim=0)[0]
        first_key=encoder_out[0]
        #sentence_representation = x[src_tokens.eq(self.encoder.dictionary.eos()), :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        
        #src_tokens bsz x seqlen
     
        
        assert (src_tokens.size()==(bsz,seqlen))
        
        #src_tokens.transpose(0, 1) 
        x_tmp=encoder_out.transpose(0, 1) 
        # x = x.transpose(0, 1)  # T x B x C -> B x T x C in decoder output
        # however encoder_out is still T x B x C
        last_key= x_tmp[src_tokens.eq(eos), :]\
        .view(x_tmp.size(0), -1, x_tmp.size(-1))[:, -1, :]
        
        #x_tmp[src_tokens.eq(self.encoder.dictionary.eos()), :] should be bsz x seqlen x fea
        # .view(x_tmp.size(0), -1, x_tmp.size(-1))[:, -1, :] size should be  bsz, fea
        #print ('last_key = {}')
        assert (last_key.size()==(bsz, fea))
        
        
        #dym_keys=[first_key,mean_key,max_key,last_key]
        if fp16==True:
            dym_keys=[first_key.half(),mean_key.half(),max_key.half(),last_key.half()]
        else:
            dym_keys=[first_key,mean_key,max_key,last_key]
        
        dym_keys=torch.stack(dym_keys,0)
        dym_keys=dym_keys.cuda()

        #========================================================================================================================
        return dym_keys

    def forward(self, encoder_out,encoder_padding_mask,src_tokens,dym_keys, **kwargs):
        '''
        x, layer_attn, _ = layer(
            x,
            encoder_state,
            encoder_padding_mask_tmp,
            incremental_state,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_attn=bool((idx == alignment_layer)),
            need_head_weights=bool((idx == alignment_layer)),
        )
        inner_states.append(x)
        '''
        #features = encoder_in.mean(0)
        
        #output length should be query length,  K and V is encoder in
        #Q is decoder_in,  decoder_in is x here
        
        '''
        sentence_representation = x[
                src_tokens.eq(self.encoder.dictionary.eos()), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            x = self.classification_heads[classification_head_name](
                sentence_representation
            )
        '''
        seqlen,bsz,fea = encoder_out.size() #liwei mod
        encoder_out=encoder_out.cuda()
        
        assert (encoder_out.size()==(seqlen,bsz,fea) and encoder_padding_mask.size()==(bsz,seqlen))

       

        #print ('first_key {} mean_key {} max_key {} last_key {}'.format(first_key.size(),mean_key.size(),max_key.size(),last_key.size()))
        qurry_keys=self.combine_keys(dym_keys,bsz)
        qurry_keys=qurry_keys.cuda()
        #print ('qurry_keys.size() = {} self.key_dim = {}'.format(qurry_keys.size(),self.key_dim))
        assert (qurry_keys.size()==(dym_keys.size()[0]+4,bsz,self.key_dim))
        
        
        master_keys=self.master_key.expand(1,bsz,self.key_dim) #(1,bsz,fea)
        master_keys=master_keys.cuda()
        assert (master_keys.size()==(1,bsz,self.key_dim))
        #==================================
        
        #decoder in here is qurry_keys
        #encoder in here is from the encoder input below
        
        #print ('\n\nqurry_keys.size() = {} master_keys.size() = {}\n\n'.format(qurry_keys.size(),master_keys.size()))
        #decoder layer :first self attention on query itself, then K,V = encoder_out Q= query 
        #====================================
        x_new, layer_attn, _ = self.keys_layer(
            query=qurry_keys,
            encoder_out=encoder_out,
            encoder_padding_mask=encoder_padding_mask, 
            incremental_state=None, #for decoder future masking, increment decode
            self_attn_mask=None, #similar
            self_attn_padding_mask=None #decoder input masking
        )
        #8 key output should 8xbaz*fea

        #print ('x_new 1.size() = {} self.key_dim = {}'.format(x_new.size(),self.key_dim))
        assert (x_new.size()==(dym_keys.size()[0]+4,bsz,fea))
        #dym_keys=torch.cat((dym_keys0, dym_keys1), dim=0).cuda()
        #=====================================================
        
        assert (dym_keys.size()==(dym_keys.size()[0],bsz,fea))
        x_new=torch.cat((x_new,dym_keys),dim=0).cuda()
        
        assert (x_new.size()==(dym_keys.size()[0]*2+4,bsz,fea))
        
        #=====================================================
        #def forward(self, encoder_out, encoder_padding_mask, attn_mask: Optional[Tensor] = None, query=None):
        #
        #sum 8 keys into one master key
        x_new= self.master_layer(
            encoder_out=x_new, #key output is the K and V
            encoder_padding_mask=None, #all 8 keys are full, no mask
            attn_mask=None, #similar
            query=master_keys  #master key is the querry now
        )
       
        #print ('x_new 2.size() = {} self.key_dim = {}'.format(x_new.size(),self.key_dim))
        
        '''
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        '''
        return x_new
        
    def build_decoder_layer(self, args, no_encoder_attn=False):
        #liwei add
        #args.encoder_embed_dim=self.encoder_embed_dim_new
        return TransformerDecoderLayer_lw_mod2(args, no_encoder_attn, lang_embedding=False)

    def build_encoder_layer(self, args, no_encoder_attn=False):
        #liwei add
        #args.encoder_embed_dim=self.encoder_embed_dim_new
        return TransformerEncoderLayer_lw_mod2(args)    
    
class BARTClassificationHead_lw(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model("transformer_lw")
class TransformerModel_lw(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        
        self.sentemb_mod=args.sentemb_mod
        
        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        #if getattr(args, 'apply_bert_init', False):
        if  args.init_mod=='BERT':
            self.apply(init_bert_params)
            
        #if args.criterion.find('_dist')>0:
        #    self.compute_dist=compute_dist(beta=args.beta,Ns=args.Ns,norm_tag=args.norm_tag,margin=args.margin,la=args.la, options=args.dist_opt)
        #else:
        #    self.compute_dist=None
            
        

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        parser.add_argument('--lang-embedding-size', type=int, default=32, metavar='N',
                            help='language embedding dimension')

        parser.add_argument('--sentemb-mod', type=str, default='empty', metavar='STR',
                            help='sentemb-mod mean, max and first')
                            

        parser.add_argument('--init-mod', type=str, default='BERT',metavar='STR',
                            help='different init setting')

        parser.add_argument('--model-type, type=int', default=0, metavar='N',
                            help='different model structures')
                            
     
                            
                            
        # fmt: on
        # fmt: on
   
        
    
        # head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
        # num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
        # inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

        # if getattr(self.args, 'load_checkpoint_heads', False):
            # if head_name not in current_head_names:
                # self.register_classification_head(head_name, num_classes, inner_dim)
        # else:
            # if head_name not in current_head_names:
                # logger.warning(
                    # 'deleting classification head ({}) from checkpoint '
                    # 'not present in current model: {}'.format(head_name, k)
                # )
                # keys_to_delete.append(k)
            # elif (
                # num_classes != self.classification_heads[head_name].out_proj.out_features
                # or inner_dim != self.classification_heads[head_name].dense.out_features
            # ):
                # logger.warning(
                    # 'deleting classification head ({}) from checkpoint '
                    # 'with different dimensions than current model: {}'.format(head_name, k)
                # )
                # keys_to_delete.append(k)
    
        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # Languages index: lang codes into integers
        lang_dictionary = {
            task.langs[i] : i for i in range(len(task.langs))
        }

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens,lang_dictionary=lang_dictionary,lang_embedding_size=args.lang_embedding_size)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder_lw(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens,lang_dictionary,lang_embedding_size):
        return TransformerDecoder_lw(
            args,
            tgt_dict,
            embed_tokens,
            lang_dictionary,lang_embedding_size,
            no_encoder_attn=getattr(args, "no_cross_attention", False)
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    '''
    laser_lstm forward()
    def forward(self, src_tokens, src_lengths, prev_output_tokens, decoder_lang, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        #encoder_sentemb = encoder_out['sentemb']

        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, lang=decoder_lang, **kwargs)
        return decoder_out

    '''
    @classmethod
    def encoder_out_proc(encoder_out_std):
        return encoder_out_std

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        decoder_lang,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        #print ('\n\ncheck alignment_layer = {}  = {}\n\n'.format(alignment_layer,alignment_heads))
        #assert(0==1)
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        
        '''
        EncoderOut = NamedTuple(
            "EncoderOut",
            [
            ("encoder_out", Tensor),  # T x B x C
            ("encoder_padding_mask", Tensor),  # B x T
            ("encoder_embedding", Tensor),  # B x T x C
            ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
            ],
        )
        '''
        #print ('\n\n\n\n\n\n\n ####### enter mdoel forward ###############\n\n\n\n\n\n\n\n\n\n')
        #print ('\n\n\n\liwei mark0 ##############\n\n\n')
        encoder_out_etd = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            cls_input=cls_input,
            return_all_hiddens=return_all_hiddens,
        )

        encoder_out=encoder_out_etd['EncoderOut']
        sentemb=encoder_out_etd['sentemb']
        
        #---------------------------------------
        #x=encoder_out.encoder_out
        #sentemb = x.mean(dim=0)[0]
        #sentemb = x.max(dim=0)[0]
        #---------------------------------------
        #print ('\n\n\n\liwei mark1 ############## encoder_out ={} \n\n\n'.format(encoder_out))
        #sentemb=etc['sentemb']
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            decoder_lang=decoder_lang,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            sentemb=sentemb
        )
        return decoder_out

    #====================================================
    # def forward(
        # self,
        # src_tokens,
        # src_lengths,
        # prev_output_tokens,
        # decoder_lang,
        # cls_input: Optional[Tensor] = None,
        # return_all_hiddens: bool = True,
        # features_only: bool = False,
        # alignment_layer: Optional[int] = None,
        # alignment_heads: Optional[int] = None,
        # tgt_tokens=None,
        # tgt_lengths=None,
        # compute_mod='compute_dist'
    # ):
        # #print ('\n\ncheck alignment_layer = {}  = {}\n\n'.format(alignment_layer,alignment_heads))
        # #assert(0==1)
        # """
        # Run the forward pass for an encoder-decoder model.
        # Copied from the base class, but without ``**kwargs``,
        # which are not supported by TorchScript.
        # """
        
        # '''
        # EncoderOut = NamedTuple(
            # "EncoderOut",
            # [
            # ("encoder_out", Tensor),  # T x B x C
            # ("encoder_padding_mask", Tensor),  # B x T
            # ("encoder_embedding", Tensor),  # B x T x C
            # ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
            # ],
        # )
        # '''
        # #print ('\n\n\n\n\n\n\n ####### enter mdoel forward ###############\n\n\n\n\n\n\n\n\n\n')
        # #print ('\n\n\n\liwei mark0 ##############\n\n\n')
        # encoder_out_etd = self.encoder(
            # src_tokens,
            # src_lengths=src_lengths,
            # cls_input=cls_input,
            # return_all_hiddens=return_all_hiddens,
        # )

        # encoder_out=encoder_out_etd['EncoderOut']
        # sentemb=encoder_out_etd['sentemb']
        
        # #---------------------------------------
        # #x=encoder_out.encoder_out
        # #sentemb = x.mean(dim=0)[0]
        # #sentemb = x.max(dim=0)[0]
        # #---------------------------------------
        # #print ('\n\n\n\liwei mark1 ############## encoder_out ={} \n\n\n'.format(encoder_out))
        # #sentemb=etc['sentemb']
        
        # if compute_mod=='compute_dist':
            # assert (tgt_tokens!=None)
            # assert (tgt_lengths!=None)
            # tgt_encoder_out_etd = self.encoder(
                # src_tokens=tgt_tokens,
                # src_lengths=tgt_lengths,
                # cls_input=None,
                # return_all_hiddens=False,
                # )

            # tgt_encoder_out=tgt_encoder_out_etd['EncoderOut']
            # tgt_sentemb=tgt_encoder_out_etd['sentemb']
        
            # Dist_M_sum,etc_dic=self.compute_dist(sentemb, tgt_sentemb)
        
        
        
        # decoder_out = self.decoder(
            # prev_output_tokens,
            # encoder_out=encoder_out,
            # decoder_lang=decoder_lang,
            # features_only=features_only,
            # alignment_layer=alignment_layer,
            # alignment_heads=alignment_heads,
            # src_lengths=src_lengths,
            # return_all_hiddens=return_all_hiddens,
            # sentemb=sentemb
        # )
        
        # if compute_mod=='compute_dist':
            # #return Dist_M_sum,
            # new_etc_dic={}
            # Dist_M_sum=Dist_M_sum.clone()
            # for key in etc_dic.keys(): 
                # new_etc_dic[key]=etc_dic[key].clone()
            
            # return decoder_out,Dist_M_sum,new_etc_dic
            # #return decoder_out,Dist_M_sum,etc_dic
            
        # return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

@register_model("transformer_align_lw")
class TransformerAlignModel_lw(TransformerModel_lw):
    """
    See "Jointly Learning to Align and Translate with Transformer
    Models" (Garg et al., EMNLP 2019).
    """

    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        self.full_context_alignment = args.full_context_alignment

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(TransformerAlignModel, TransformerAlignModel).add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='D',
                            help='Number of cross attention heads per layer to supervised with alignments')
        parser.add_argument('--alignment-layer', type=int, metavar='D',
                            help='Layer number which has to be supervised. 0 corresponding to the bottommost layer.')
        parser.add_argument('--full-context-alignment', type=bool, metavar='D',
                            help='Whether or not alignment is supervised conditioned on the full target context.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        transformer_align(args)

        transformer_model = TransformerModel.build_model(args, task)
        return TransformerAlignModel(
            transformer_model.encoder, transformer_model.decoder, args
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        return self.forward_decoder(prev_output_tokens, encoder_out)

    def forward_decoder(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        attn_args = {
            "alignment_layer": self.alignment_layer,
            "alignment_heads": self.alignment_heads,
        }
        decoder_out = self.decoder(prev_output_tokens, encoder_out, **attn_args)

        if self.full_context_alignment:
            attn_args["full_context_alignment"] = self.full_context_alignment
            _, alignment_out = self.decoder(
                prev_output_tokens,
                encoder_out,
                features_only=True,
                **attn_args,
                **extra_args,
            )
            decoder_out[1]["attn"] = alignment_out["attn"]

        return decoder_out


class TransformerEncoder_lw(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)

        self.sentemb_mod=args.sentemb_mod

        self.register_buffer("version", torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        
        self.fp16=args.fp16
       
        if args.test_mod =='fp32':
            self.fp16=False
        print ('\nliwei args.test_mod = {} self.fp16 = {}\n'.format(args.test_mod,self.fp16))
        #assert(0==1)
        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None
            
        #=================================================
        #liwei mod
        
        #if self.sentemb_mod=='keymaster0':
        if self.sentemb_mod.find('keymaster0')>=0:
            key_dim_tmp=args.encoder_embed_dim

            self.keymaster0=key_master(args,input_dim =args.encoder_embed_dim,\
            key_dim=key_dim_tmp,output_dim=key_dim_tmp,eos=self.dictionary.eos(),\
            activation_fn="tanh",pooler_dropout=0.0)


            #==================================================================================
            #self.keymaster1=key_master(args,input_dim =args.encoder_embed_dim,\
            #key_dim=key_dim_tmp,output_dim=key_dim_tmp,eos=self.dictionary.eos(),\
            #activation_fn="tanh",pooler_dropout=0.0)
            
            #self.sentemb_layer_lw=sentemb_layer_lw(input_dim=key_dim_tmp*2,inner_dim=key_dim_tmp,\
            #output_dim=key_dim_tmp,activation_fn="tanh",pooler_dropout=0.0)
            #=========================================================================================
            
            #self.layers.extend([self.keymaster0,self.keymaster1,self.sentemb_layer_lw])
            
        
        #
        '''
        def __init__(
        self,
        args,
        input_dim,
        key_dim,
        output_dim,
        eos,
        activation_fn="tanh",
        pooler_dropout=0.0,
        '''
        
        #sentemb_layer_lw
        '''
        def __init__(
            self,
            input_dim,
            inner_dim,
            output_dim,
            activation_fn,
            pooler_dropout,
        ):
        '''
        
    #@staticmethod
    #def add_args(parser):
         
        

    def build_encoder_layer(self, args):
        return TransformerEncoderLayer_lw(args)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    #def forward(self, src_tokens, src_lengths): laser encoder forward
    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

       
        
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)



        #tmp_x=x

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        seqlen,bsz,fea = x.size() #liwei mod

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        #=====================================================
        '''
        print ('self.padding_idx = {}, tmp_x ={} encoder_padding_mask = {} encoder_embedding ={} src_tokens = {}, src_tokens.size = {}  src_lengths = {} cls_input = {}'\
            .format(self.padding_idx,tmp_x, encoder_padding_mask,encoder_embedding, src_tokens,src_tokens.size(),src_lengths,cls_input))
        #assert(0==1)

        print ('tmp_x.size = {} src_lengths ={}'.format(tmp_x.size(),src_lengths.size()))
        encoder_padding_mask_tmp=encoder_padding_mask.transpose(0,1)
        mean_tmp1=x.masked_fill_(encoder_padding_mask_tmp.unsqueeze(-1), 0.0).sum(dim = 0)/((src_lengths +1).unsqueeze(-1))
        #mean_tmp1=tmp_x.mean(dim = 1)
        mean_tmp2=tmp_x.sum(dim = 1)/((src_lengths +1).unsqueeze(-1))
        mean_tmp3=tmp_x.masked_fill_(encoder_padding_mask.unsqueeze(-1), 0.0).sum(dim = 1)/((src_lengths +1).unsqueeze(-1))
        #mean_tmp4=tmp_x[0:src_lengths,:].mean()
        mean_tmp4=tmp_x.sum(dim = 1)
        mean_tmp5=tmp_x.masked_fill_(encoder_padding_mask.unsqueeze(-1), 0.0).sum(dim = 1)
        print ('mean = {} \n######\n {} \n#####\n {} \n######\n {} \n ####### \n {}'.format(mean_tmp1,mean_tmp2,mean_tmp3,mean_tmp4,mean_tmp5))

        assert (0==1)
        '''
        #======================================================

        encoder_states = [] if return_all_hiddens else None

        layer_count=0
        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                layer_count+=1
                if (layer_count==4):
                    encoder_states_4=x
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)


        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        #==============================        
        #x=encoder_out.encoder_out
        #sentemb = x.mean(dim=0)[0]
        #self.layers.extend([self.keymaster0,self.keymaster1,sentemb_layer_lw])
        #def forward(self, encoder_out,encoder_padding_mask,src_tokens, **kwargs):
        
        #if self.sentemb_mod=='keymaster0':
        if self.sentemb_mod.find('keymaster0')>=0:

            dym_keys0=key_master.get_dym_keys(encoder_out=x, encoder_padding_mask=encoder_padding_mask,src_tokens=src_tokens,eos=self.dictionary.eos(),fp16=self.fp16)
            
            #print ('dym_keys0 {}'.format(dym_keys0))

            #print ('encoder_states {} encoder_padding_mask{}'.format(encoder_states,encoder_padding_mask))

            #===============================================================================
            if self.sentemb_mod.find('keymaster08')>=0:
                dym_keys1=key_master.get_dym_keys(encoder_out=encoder_states_4, encoder_padding_mask=encoder_padding_mask,src_tokens=src_tokens,eos=self.dictionary.eos(),fp16=self.fp16)
                dym_keys=torch.cat((dym_keys0, dym_keys1), dim=0).cuda()
                assert( dym_keys.size() == (8,bsz,fea))
            elif self.sentemb_mod.find('keymaster04')>=0:
                dym_keys=dym_keys0
                assert( dym_keys.size() == (4,bsz,fea))
            else:
                dym_keys1=key_master.get_dym_keys(encoder_out=encoder_states_4, encoder_padding_mask=encoder_padding_mask,src_tokens=src_tokens,eos=self.dictionary.eos(),fp16=self.fp16)
                dym_keys=torch.cat((dym_keys0, dym_keys1), dim=0).cuda()
                assert( dym_keys.size() == (8,bsz,fea))
                
            

            sentemb0=self.keymaster0(encoder_out=x, encoder_padding_mask=encoder_padding_mask,src_tokens=src_tokens,dym_keys=dym_keys)
            #print('\nsentemb0 = {}\n'.format(sentemb0.size()))
            sentemb0=sentemb0.view(bsz,fea).cuda()
            sentemb=sentemb0
            #==============================================================
            # sentemb1=self.keymaster1(encoder_out=encoder_states_4, encoder_padding_mask=encoder_padding_mask,src_tokens=src_tokens,dym_keys=dym_keys)
            # sentemb1=sentemb1.view(bsz,fea).cuda()

            # #print('\nsentemb1 = {}\n'.format(sentemb1.size()))

            # assert (sentemb0.size()==(bsz,fea))

            # sentemb=torch.cat((sentemb0, sentemb1), dim=-1).cuda()
            # assert (sentemb.size()==(bsz,fea*2))

            # sentemb=self.sentemb_layer_lw(sentemb)
            #======================================================================
            
            assert (sentemb.size()==(bsz,fea))

        elif self.sentemb_mod=='max':
            sentemb = x.max(dim=0)[0]  #max(dim) return is tuple (max vector, indics)
        elif self.sentemb_mod=='mean':
            #sentemb = x.mean(dim=0)
            #mask_mean(tmp_x,padding_mask, dim, src_lengths)
            #print ('x.size= {}, encoder_padding_mask size ={} ,seqlen,bsz,fea ={}'.format())

            #src_lengths_tmp=(encoder_padding_mask<=0).sum(dim=1)
            #print ('src_lengths_tmp = {} src_lengths ={}'.format(src_lengths_tmp,src_lengths))
            #assert (torch.equal(src_lengths_tmp, src_lengths))

            assert (x.size()==(seqlen,bsz,fea) and encoder_padding_mask.size()==(bsz,seqlen))
            sentemb=mask_mean(tmp_x=x,padding_mask=encoder_padding_mask.transpose(0,1),dim=0,src_lengths=None)

        elif self.sentemb_mod=='first':
            sentemb = x[0]
        else:
            print ('need to set sentemb_mod')
            assert(0==1)

        assert (sentemb.size()==(bsz,fea))
        sentemb=sentemb.cuda()
        #==============================
        
        return {'EncoderOut':EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        ),'sentemb':sentemb}
        
        return out
    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder_lw(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, lang_dictionary,lang_embedding_size=32, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.lang_embedding_size = lang_embedding_size
        self.lang_dictionary = lang_dictionary
        self.embed_langs = nn.Embedding(len(lang_dictionary), lang_embedding_size)

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        #self.encoder_embed_dim_decatt=args.encoder_embed_dim+arg.lang_embedding_size


        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        print ('\n\n\n\n\n\n\n\n\n\n\n############ input_embed_dim = {}, embed_dim = {}, self.output_embed_dim ={} args ={}###############\n\n\n\n\n\n\n\n\n\n\n\n\n'
            .format(input_embed_dim,embed_dim,self.output_embed_dim,args))

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None


        
    def build_decoder_layer(self, args, no_encoder_attn=False):
        #liwei add
        #args.encoder_embed_dim=self.encoder_embed_dim_new
        return TransformerDecoderLayer_lw(args, no_encoder_attn ,lang_embedding=True)

    #forward(self, prev_output_tokens, encoder_out, lang, incremental_state=None): laser decoder forward

    #encoder_out: Optional[EncoderOut] = None,
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        decoder_lang = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        sentemb: Optional[Any] = None,
    ):
        #print ('\n\n\n\n\n\n\n enter decoder forward \n\n\n\n\n\n\n\n\n\n')
        #print ('\n\n\n\n\n\n\n  decoder forward encoder_out = {} \n\n\n\n\n\n\n\n\n\n'.format(encoder_out))
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        
        x, extra = self.extract_features(
            prev_output_tokens,
            sentemb=sentemb,
            encoder_out=encoder_out,
            decoder_lang=decoder_lang,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra,{'sentemb':sentemb}
    '''
    laser_lstm:
    def extract_features(
        self, prev_output_tokens, encoder_out, lang, incremental_state=None
    '''
    def extract_features(
        self,
        prev_output_tokens,
        sentemb,
        encoder_out: Optional[EncoderOut] = None,
        decoder_lang = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        #print ('\n\n\n\n\n\n\n  extract_features forward encoder_out = {} \n\n\n\n\n\n\n\n\n\n'.format(encoder_out))
        #print('encoder_out = {}'.format(encoder_out[0]))
        #encoder_out=encoder_out[0]

       
        bsz, seqlen = prev_output_tokens.size() #liwei mod

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1


        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        #=======================================================
        #liwei mod
        
        # embed language
        lang_tensor = torch.LongTensor(
            [self.lang_dictionary[decoder_lang]] * bsz
        ).to(device=prev_output_tokens.device)
        l = self.embed_langs(lang_tensor)
        encoder_sentemb=sentemb

            #liwei mod, compile encoder_out to 1 single encoder_sentemb
        #===================================================================================================================================================
        #print ('\n bsz = {} seqlen= {} x size = {} encoder_sentemb size = {} l size = {} encoder_out ={}\n'
        #    .format(bsz,seqlen,x.size(),encoder_sentemb.size(),l.size(),encoder_out.encoder_out.size()))
       
        encoder_sentemb_tmp= torch.cat((encoder_sentemb, l), dim=-1)
        encoder_state_tmp=encoder_sentemb_tmp.view(1,bsz,encoder_sentemb_tmp.size()[-1])
        encoder_state_tmp=encoder_state_tmp.cuda()
        
        #print ('\nafter bsz = {} seqlen= {} x size = {} encoder_sentemb size = {} l size = {} encoder_out ={}\n'
        #    .format(bsz,seqlen,x.size(),encoder_sentemb.size(),l.size(),encoder_out.encoder_out.size()))
        #print ('\nafter bsz = {} seqlen= {} x size = {} encoder_sentemb_tmp size = {} l size = {} encoder_out ={} encoder_state_tmp = {}\n'
        #    .format(bsz,seqlen,x.size(),encoder_sentemb_tmp.size(),l.size(),encoder_out.encoder_out.size(),encoder_state_tmp.size()))
        #assertTrue(sentemb.size()[1]==bsz)
        
        
        #assert(0==1)

        encoder_padding_mask_tmp = encoder_out.encoder_padding_mask.new_zeros(encoder_state_tmp.size(1), encoder_state_tmp.size(0))
        encoder_padding_mask_tmp=encoder_padding_mask_tmp.cuda()

        
        #print ('encoder_padding_mask = {} encoder_padding_mask size= {} encoder_padding_mask_tmp ={} encoder_padding_mask_tmp size={}'
        #    .format(encoder_out.encoder_padding_mask,encoder_out.encoder_padding_mask.size(),encoder_padding_mask_tmp,encoder_padding_mask_tmp.size()))
        #assert (1==0)

        #==================================================================================================================================================




        #==============================================================
        '''
        encoder_sentemb_tmp1=encoder_sentemb.expand(seqlen,-1,-1)
        l=l.expand(seqlen,-1,-1)
        input = torch.cat((x, encoder_sentemb_tmp1, l), dim=-1)
        print ('\nx size = {} encoder_sentemb size = {} l size ={}, input size = {}\n'.format(x.size(),encoder_sentemb.size(),l.size(),input.size()))
        x=input.cuda()
        '''
        #==============================================================

        '''
        encoder_out_tmp=encoder_out.encoder_out
        encoder_sentemb_tmp2=encoder_sentemb.expand(encoder_out_tmp.size()[0],-1,-1)
        
        print ('\n bsz = {} seqlen= {} encoder_out_tmp size = {} encoder_sentemb size = {}'.format(bsz,seqlen,encoder_out_tmp.size(),encoder_sentemb_tmp2.size()))
        encoder_out_tmp=torch.cat((encoder_out_tmp,encoder_sentemb_tmp2), dim=-1)
        print ('\n after bsz = {} seqlen= {} encoder_out_tmp size = {} encoder_sentemb size = {}'.format(bsz,seqlen,encoder_out_tmp.size(),encoder_sentemb_tmp2.size()))

        print ('\nx size = {} encoder_sentemb size = {} l size ={}, input size = {}\n'.format(x.size(),encoder_sentemb.size(),l.size(),input.size()))
        '''

        #print ('\nenumerate self.layers = {}\n'.format(enumerate(self.layers)))
        #=======================================================
        '''
        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_sentemb for i in range(num_layers)]
            prev_cells = [encoder_sentemb for i in range(num_layers)]
            prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
            prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size) #init input at zeros

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], encoder_sentemb, input_feed, l), dim=1)
            ########
            input_feed = out
        '''
        assert (self.cross_self_attention==False)

        self_attn_padding_mask: Optional[Tensor] = None
        #print ('\n\n\n\n\n\nself_attn_padding_mask  {}\n\n\n\n\n\n'.format(self_attn_padding_mask))
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        #print ('\n\n\n\n\n\nself_attn_padding_mask  {}\n\n\n\n\n\n'.format(self_attn_padding_mask))
        #assert (self_attn_padding_mask==None)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            #=============================================
            #liwei mod comment out make encoder state always = sentemb
            '''
            encoder_state: Optional[Tensor] = None

            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                    #------------------------------
                    print ('go to the layer_wise_attention, wrong path')
                    assert (1==0)
                else:
                    encoder_state = encoder_out.encoder_out
             '''       
            #============================================================

            encoder_state=encoder_state_tmp

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            #=======================================
            #
            '''
             encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
            '''
            #liwei change padding to None
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state,
                    encoder_padding_mask_tmp,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]


    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("transformer_lw", "transformer_lw")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)


@register_model_architecture("transformer_lw", "transformer_lw_dec1")
def base_architecture_dec1(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)   


@register_model_architecture("transformer_lw", "transformer_lw_dec3")
def base_architecture_dec3(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)     
    
    
# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod1024X4096X16dec1")
def transformer_iwslt_de_en_lw_mod1024X4096X16dec1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    #args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture_dec1(args)

@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod")
def transformer_iwslt_de_en_lw_mod(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod_dec1")
def transformer_iwslt_de_en_lw_mod_dec1(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 1)
    base_architecture(args)    
    


@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod1024X4096X16")
def transformer_iwslt_de_en_lw_mod1024X4096X16(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod1024X4096X16enc")
def transformer_iwslt_de_en_lw_mod1024X4096X16enc(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw_mod1024x2048x8")
def transformer_iwslt_de_en_lw_mod1024x2048x8(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)



 



@register_model_architecture("transformer_lw", "transformer_iwslt_de_en_lw")
def transformer_iwslt_de_en_lw(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)


@register_model_architecture("transformer_lw", "transformer_wmt_en_de_lw")
def transformer_wmt_en_de_lw(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_de_big_norm_lw")
def transformer_vaswani_wmt_en_de_big_norm_lw(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    #args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_de_big_lw")
def transformer_vaswani_wmt_en_de_big_lw(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)
    
# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_de_big_lw_dec3")
def transformer_vaswani_wmt_en_de_big_lw_dec3(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture_dec3(args)    
    
    
# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_de_bigmid_lw")
def transformer_vaswani_wmt_en_de_bigmid_lw(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)   
    
    
# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_de_bigmid_lw_dec3")
def transformer_vaswani_wmt_en_de_bigmid_lw_dec3(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture_dec3(args)    
    


@register_model_architecture("transformer_lw", "transformer_vaswani_wmt_en_fr_big_lw")
def transformer_vaswani_wmt_en_fr_big_lw(args):
    args.dropout = getattr(args, "dropout", 0.1)
    transformer_vaswani_wmt_en_de_big_lw(args)


@register_model_architecture("transformer_lw", "transformer_wmt_en_de_big_lw")
def transformer_wmt_en_de_big_lw(args):
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big_lw(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_lw", "transformer_wmt_en_de_big_t2t_lw")
def transformer_wmt_en_de_big_t2t_lw(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big_lw(args)
    
# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_lw", "transformer_wmt_en_de_big_t2t_lw_dec3")
def transformer_wmt_en_de_big_t2t_lw(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_big_lw_dec3(args)    
    
    
# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_lw", "transformer_wmt_en_de_bigmid_t2t_lw")
def transformer_wmt_en_de_bigmid_t2t_lw(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_bigmid_lw(args)   
    
# default parameters used in tensor2tensor implementation
@register_model_architecture("transformer_lw", "transformer_wmt_en_de_bigmid_t2t_lw_dec3")
def transformer_wmt_en_de_bigmid_t2t_lw_dec3(args):
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    transformer_vaswani_wmt_en_de_bigmid_lw_dec3(args)   
    
    

    


# @register_model_architecture("transformer_align", "transformer_align")
# def transformer_align(args):
#     args.alignment_heads = getattr(args, "alignment_heads", 1)
#     args.alignment_layer = getattr(args, "alignment_layer", 4)
#     args.full_context_alignment = getattr(args, "full_context_alignment", False)
#     base_architecture(args)


# @register_model_architecture("transformer_align", "transformer_wmt_en_de_big_align")
# def transformer_wmt_en_de_big_align(args):
#     args.alignment_heads = getattr(args, "alignment_heads", 1)
#     args.alignment_layer = getattr(args, "alignment_layer", 4)
#     transformer_wmt_en_de_big(args)