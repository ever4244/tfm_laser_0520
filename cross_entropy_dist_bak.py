# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch.nn as nn

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

import numpy as np
import torch
from statistics import mean

import timeit

     
        # #torch.cuda.current_device()

        # #print ('enter CrossEntropyCriterion_dist forward\n')
        
        # #sample[net_input].keys() = dict_keys(['src_tokens', 'src_lengths', 'prev_output_tokens', 'decoder_lang'])
        # #sample[target].size() = torch.Size([12, 49])

        

        # #tgt_emb = model.encoder(tgt_tokens, tgt_lengths)['sentemb']
        # #tar_emb=model.LaserEncoder.forward(sample['target'],sample['target'].size()[1])

        # #print ('sample = {}\n sample[net_input] ={} \n sample.keys() = {}\n sample[net_input].keys() = {}\n sample[target].size() = {}\n'.format(sample, sample['net_input'], sample.keys(), sample['net_input'].keys(),sample['target'].size()))
        # #print ('sample[id] = {}'.format(sample['id']))

        
        # #src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        # #src_lengths, sort_order = src_lengths.sort(descending=True)
        # #tgt_lengths= torch.LongTensor([s['source'].numel() for s in samples])
        
        # #print ('src_lengths = {}\n, src_tokens = {}\n. tgt_lengths = {}\n, tgt_tokens ={}\n, tgt_tokens.size()[1] = {}\n'.format(sample['net_input']['src_lengths'],sample['net_input']['src_tokens'].size(),tgt_lengths,tgt_tokens.size(),tgt_tokens.size()[1]))

        # ''' src sort example
        # src_lengths, sort_order = src_lengths.sort(descending=True)
        # id = id.index_select(0, sort_order)
        # src_tokens = src_tokens.index_select(0, sort_order)
        # '''

        # #arcsort example
        # '''
        # start_id = 0
        # for inputs in buffered_read(args.input, args.buffer_size):
        # indices = []
        # results = []
        # for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            # src_tokens = batch.src_tokens
            # src_lengths = batch.src_lengths
            # if use_cuda:
                # src_tokens = src_tokens.cuda()
                # src_lengths = src_lengths.cuda()
            
            # model.eval()
            # embeddings = model.encoder(src_tokens, src_lengths)['sentemb']
            # embeddings = embeddings.detach().cpu().numpy()
            # for i, (id, emb) in enumerate(zip(batch.ids.tolist(), embeddings)):
                # indices.append(id)
                # results.append(emb)
        # np.vstack(results)[np.argsort(indices)].tofile(fout)

        # # update running id counter
        # start_id += len(inputs)
        # fout.close()
        # '''
        # #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # #tgt_order_id=torch.LongTensor(np.arange(tgt_lengths.size()[0]))
        # #tgt_order_id=tgt_order_id.index_select(0, tgt_sort_order)
        # #print('tgt_order_id = {}'.format(tgt_order_id))
       
        # #print ('presorted tgt_lengths = {}'.format(tgt_lengths))
        
        # #print ('sorted tgt_lengths = {}'.format(tgt_lengths))

        
        
        # #print ('net_output ={}\n net_output[0].size() = {}\n net_output[2].size() = {}\n'.format(net_output,net_output[0].size(), net_output[2].size()))
        # #sample[net_output].keys() = {}\n ,sample['net_output'].keys()

        # #laser decoder forward :
        # '''
            # x, attn_scores = self.extract_features(
                # prev_output_tokens, encoder_out, lang, incremental_state
            # )
            # #return self.output_layer(x), attn_scores 
            # encoder_sentemb = encoder_out['sentemb']#liwei mod
            # return self.output_layer(x), attn_scores, encoder_sentemb
        # '''

        # '''
            # transformer decoder forward;
             # x, extra = self.extract_features(
            # prev_output_tokens,
            # encoder_out=encoder_out,
            # incremental_state=incremental_state,
            # alignment_layer=alignment_layer,
            # alignment_heads=alignment_heads,
            # )
            # if not features_only:
                # x = self.output_layer(x)
            # return x, extra
        # '''


import torch.nn.functional as F


class compute_dist(nn.Module):
    
    def __init__(self,beta,Ns,norm_tag,margin,la,options=None):
        super().__init__()
        self.la=la
        #beta=0.1 #current best
        self.beta=beta
        self.Ns=Ns
        
        self.timers={'rec0':0, 'rec1':0, 'rec2':0, 'rec3':0, 'rec4':0}
        #self.norm_tag='nuc'
        self.norm_tag=norm_tag
        self.options=options
        self.margin=margin
        

        print ('\n###transformer beta = {}, Ns = {}, norm_tag = {} margin = {} self.la={} self.options = {}###\n'.format(self.beta,self.Ns,self.norm_tag,self.margin,self.la, self.options))
       
      
      
    # @staticmethod
    # def add_args(parser):
        # ===================================================================
        # parser.add_argument('--margin', default=0.5, type=float, metavar='D',
                            # help='margin for dist constraint')

        # parser.add_argument('--beta', default=0.2, type=float, metavar='D',
                            # help='beta for dist constraint')
                            
        # parser.add_argument('--la', default=0.5, type=float, metavar='D',
                            # help='lambda for dist constraint')

        # parser.add_argument('--Ns', default=20, type=int, metavar='N',
                            # help='beta for dist constraint')

        # parser.add_argument('--dist-opt', type=str, default=None, metavar='STR',
                            # help='different dist constraint: cos dnorm')
        # ======================================================================
    
    
    def norm_lw(self, vec):
        
        vec=vec.cuda()
        #vec_norm=torch.sqrt(vec.pow(2)).sum(-1)
        #if vec.dim()==1:
        #    vec_norm=torch.sqrt(vec.pow(2)).sum(0)
        #else:
        #    vec_norm=torch.sqrt(vec.pow(2).sum(1))
        #
        vec_norm=vec.norm(p=self.norm_tag,dim=-1)
        return vec_norm


        
    def distance_func_cos(self, self_att_ctx1, self_att_ctx2):

        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        
        margin=self.margin

        

        vec_en2de = self_att_ctx1.cuda()
        vec_de2en = self_att_ctx2.cuda()


        vec_en2de_mean = vec_en2de.mean(dim=0).cuda()
        vec_en2de_mean_norm=self.norm_lw(vec_en2de_mean)
        vec_de2en_mean_norm=vec_en2de_mean_norm #save computation

        
        batch_size = vec_en2de.size()[0]

     
        #dist_pair_batch= self.norm_lw (vec_en2de - vec_de2en)
        dist_pair_batch=angular_distance(vec_en2de,vec_de2en)


       
        NS_num=self.Ns
        def get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j=0):
            
            #dist_enNS_batch=self.norm_lw (vec_en2de - vec_de2en[torch.randperm(batch_size)])
            dist_enNS_batch=angular_distance(vec_en2de,vec_de2en[torch.randperm(batch_size)])

            #dist_deNS_batch=self.norm_lw (vec_de2en - vec_en2de[torch.randperm(batch_size)])
            dist_deNS_batch=angular_distance (vec_de2en, vec_en2de[torch.randperm(batch_size)])
            
            
            #dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch) / (vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)).cuda()
            dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch)).cuda()


            #dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch) / (vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)).cuda()
            dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch)).cuda()


            
            dist_en=dist_en_batch.sum(0)
            dist_de=dist_de_batch.sum(0)


            #print ('\n margin = {} dist_pair_batch = {}, dist_enNS_batch = {}, dist_deNS_batch = {}\n'.format(margin, dist_pair_batch, dist_pair_batch ,dist_enNS_batch))
            #print ('\n dist_en_batch = {} dist_de_batch = {}, dist_en = {}, dist_de = {}\n'.format(dist_en_batch, dist_de_batch, dist_en ,dist_de))
            
            #print ('dist_batch = '.format(dist_en+dist_de))
            return dist_en+dist_de

        #dist_M=[]
        dist_sum=0
        for j in range(NS_num):
            dist_tmp=get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j)
            
            dist_sum=dist_sum+dist_tmp
        #dist_M=np.array(dist_M)
        etc_dic={'vec_en2de_mean_norm':vec_en2de_mean_norm,'vec_de2en_mean_norm':vec_de2en_mean_norm}
        dist_sum=dist_sum/NS_num
        #dist_sum=dist_sum.cuda()
        #print ('margin = {}'.format(margin))
        '''
        end.record()
        torch.cuda.synchronize()
        self.timers['rec1']+=start.elapsed_time(end)
        '''
        return dist_sum, etc_dic
        
    def distance_cos_plus(self, self_att_ctx1, self_att_ctx2):

        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        
        margin=self.margin
        
        vec_en2de = self_att_ctx1.cuda()
        vec_de2en = self_att_ctx2.cuda()


        vec_en2de_mean = vec_en2de.mean(dim=0).cuda()
        vec_en2de_mean_norm=self.norm_lw(vec_en2de_mean)
        vec_de2en_mean_norm=vec_en2de_mean_norm #save computation

        
        batch_size = vec_en2de.size()[0]

     
        #dist_pair_batch= self.norm_lw (vec_en2de - vec_de2en)
        dist_pair_batch=angular_distance(vec_en2de,vec_de2en)
        dist_pair_batch_sum=dist_pair_batch.sum(dim=0)


       
        NS_num=self.Ns
        def get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j=0):
            
            #dist_enNS_batch=self.norm_lw (vec_en2de - vec_de2en[torch.randperm(batch_size)])
            dist_enNS_batch=angular_distance(vec_en2de,vec_de2en[torch.randperm(batch_size)])

            #dist_deNS_batch=self.norm_lw (vec_de2en - vec_en2de[torch.randperm(batch_size)])
            dist_deNS_batch=angular_distance (vec_de2en, vec_en2de[torch.randperm(batch_size)])
            
            
            #dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch) / (vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)).cuda()
            dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch)).cuda()


            #dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch) / (vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)).cuda()
            dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch)).cuda()


            
            dist_en=dist_en_batch.sum(0)
            dist_de=dist_de_batch.sum(0)


            #print ('\n margin = {} dist_pair_batch = {}, dist_enNS_batch = {}, dist_deNS_batch = {}\n'.format(margin, dist_pair_batch, dist_pair_batch ,dist_enNS_batch))
            #print ('\n dist_en_batch = {} dist_de_batch = {}, dist_en = {}, dist_de = {}\n'.format(dist_en_batch, dist_de_batch, dist_en ,dist_de))
            
            #print ('dist_batch = '.format(dist_en+dist_de))
            return dist_en+dist_de

        #dist_M=[]
        dist_Ns_sum=0
        for j in range(NS_num):
            dist_tmp=get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j)
            
            dist_Ns_sum=dist_Ns_sum+dist_tmp
        #dist_M=np.array(dist_M)
        etc_dic={'vec_en2de_mean_norm':vec_en2de_mean_norm,'vec_de2en_mean_norm':vec_de2en_mean_norm}
        dist_Ns_sum=dist_Ns_sum/NS_num
        dist_Ns_sum=dist_Ns_sum
        
        dist_sum_final=dist_pair_batch_sum+self.la*dist_Ns_sum
        
        #print ('margin = {}'.format(margin))
        '''
        end.record()
        torch.cuda.synchronize()
        self.timers['rec1']+=start.elapsed_time(end)
        '''
        return dist_sum_final, etc_dic    

    def distance_dnorm_plus(self, self_att_ctx1, self_att_ctx2):
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        margin=self.margin

        vec_en2de = self_att_ctx1.cuda()
        vec_de2en = self_att_ctx2.cuda()
        
        batch_size = vec_en2de.size()[0]

        vec_en2de_mean = vec_en2de.mean(dim=0).cuda()
        vec_de2en_mean = vec_de2en.mean(dim=0).cuda()
        
        vec_en2de_mean_norm=self.norm_lw(vec_en2de_mean)
        vec_de2en_mean_norm=self.norm_lw(vec_de2en_mean)
        dist_pair_batch= self.norm_lw (vec_en2de - vec_de2en)
        
        
        
        vec_norm=(vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)/2
        #print ('vec_norm = {}'.format(vec_norm))
        
        #if (margin-vec_norm) <=0 or 50*vec_norm<margin :
        #    print ('margin = {} vec_norm = {}'.format(margin,vec_norm))
        
        
        dist_pair_batch_sum=dist_pair_batch.sum(dim=0)/vec_norm
        
        NS_num=self.Ns
        def get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j=0):

            
            
            dist_enNS_batch=self.norm_lw (vec_en2de - vec_de2en[torch.randperm(batch_size)])
            dist_deNS_batch=self.norm_lw (vec_de2en - vec_en2de[torch.randperm(batch_size)])
            
            #print ('dist_pair_batch = {}, dist_enNS_batch = {}, dist_deNS_batch = {}\n'.format(dist_pair_batch, dist_pair_batch ,dist_enNS_batch))
            
            dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch) / vec_norm).cuda()
            dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch) / vec_norm).cuda()
            
            dist_en=dist_en_batch.sum(dim=0)
            dist_de=dist_de_batch.sum(dim=0)
            #print ('dist_batch = '.format(dist_en+dist_de))
            return dist_en+dist_de

        #dist_M=[]
        dist_Ns_sum=0
        for j in range(NS_num):
            dist_tmp=get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j)
            
            dist_Ns_sum=dist_Ns_sum+dist_tmp
        #dist_M=np.array(dist_M)
        etc_dic={'vec_en2de_mean_norm':vec_en2de_mean_norm,'vec_de2en_mean_norm':vec_de2en_mean_norm}
        dist_Ns_sum=dist_Ns_sum/NS_num
        dist_Ns_sum=dist_Ns_sum.cuda()
        
        dist_sum_final=dist_pair_batch_sum+self.la*dist_Ns_sum
        #effectively 2X dist_pair_batch_sum 0.5X(dist_Ns_en +dist_Ns_De)
        
        #print ('margin = {}'.format(margin))
        '''
        end.record()
        torch.cuda.synchronize()
        self.timers['rec1']+=start.elapsed_time(end)
        '''
        return dist_sum_final, etc_dic
        
    def distance_func2(self, self_att_ctx1, self_att_ctx2):
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        margin=self.margin

        vec_en2de = self_att_ctx1.cuda()
        vec_de2en = self_att_ctx2.cuda()
        
        batch_size = vec_en2de.size()[0]

        vec_en2de_mean = vec_en2de.mean(dim=0).cuda()
        vec_de2en_mean = vec_de2en.mean(dim=0).cuda()
        
        vec_en2de_mean_norm=self.norm_lw(vec_en2de_mean)
        vec_de2en_mean_norm=self.norm_lw(vec_de2en_mean)
        dist_pair_batch= self.norm_lw (vec_en2de - vec_de2en)
        
        vec_norm=(vec_en2de_mean_norm+vec_de2en_mean_norm + 0.000001)/2
        #print ('vec_norm = {}'.format(vec_norm))
        
        #if (margin-vec_norm) <=0 or 50*vec_norm<margin :
        #    print ('margin = {} vec_norm = {}'.format(margin,vec_norm))
        
        NS_num=self.Ns
        def get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j=0):

            
            
            dist_enNS_batch=self.norm_lw (vec_en2de - vec_de2en[torch.randperm(batch_size)])
            dist_deNS_batch=self.norm_lw (vec_de2en - vec_en2de[torch.randperm(batch_size)])
            
            #print ('dist_pair_batch = {}, dist_enNS_batch = {}, dist_deNS_batch = {}\n'.format(dist_pair_batch, dist_pair_batch ,dist_enNS_batch))
            
            dist_en_batch = F.relu(margin + (dist_pair_batch - dist_enNS_batch) / vec_norm).cuda()
            dist_de_batch = F.relu(margin + (dist_pair_batch - dist_deNS_batch) / vec_norm).cuda()
            
            dist_en=dist_en_batch.sum(0)
            dist_de=dist_de_batch.sum(0)
            #print ('dist_batch = '.format(dist_en+dist_de))
            return dist_en+dist_de

        #dist_M=[]
        dist_sum=0
        for j in range(NS_num):
            dist_tmp=get_cost_test(dist_pair_batch,vec_en2de, vec_de2en,j)
            
            dist_sum=dist_sum+dist_tmp
        #dist_M=np.array(dist_M)
        etc_dic={'vec_en2de_mean_norm':vec_en2de_mean_norm,'vec_de2en_mean_norm':vec_de2en_mean_norm}
        dist_sum=dist_sum/NS_num
        #dist_sum=dist_sum.cuda()
        #print ('margin = {}'.format(margin))
        '''
        end.record()
        torch.cuda.synchronize()
        self.timers['rec1']+=start.elapsed_time(end)
        '''
        return dist_sum, etc_dic

    
    def forward_dist(self, src_embeddings, tgt_embeddings):
       
        
        #===========================================================================
        #options={}
        #options['dist_mode'] = 0
        
        if self.options == 'cos':
            assert(0==1)
            Dist_M_sum,etc_dic=self.distance_func_cos(src_embeddings,tgt_embeddings)
        elif self.options == 'cos+':
            Dist_M_sum,etc_dic=self.distance_cos_plus(src_embeddings,tgt_embeddings)    
        elif self.options == 'dnorm':
            assert(0==1)
            Dist_M_sum,etc_dic=self.distance_func2(src_embeddings,tgt_embeddings)
        elif self.options == 'dnorm+':
            Dist_M_sum,etc_dic=self.distance_dnorm_plus(src_embeddings,tgt_embeddings)
        else:    
            assert(0==1)

        Dist_M_sum=Dist_M_sum.cuda()
        
        return Dist_M_sum,etc_dic
    def forward(self, src_embeddings, tgt_embeddings):
        
        return self.forward_dist(src_embeddings, tgt_embeddings)
        


@register_criterion('cross_entropy_dist')
class CrossEntropyCriterion_dist(FairseqCriterion):
    
    
    
    def __init__(self, task, sentence_avg,beta=0.2,Ns=20,norm_tag='fro',margin=0.5,la=0.5,options=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        
        self.la=la
    
        #beta=0.1 #current best
        self.beta=beta
        self.Ns=Ns
        
        self.timers={'rec0':0, 'rec1':0, 'rec2':0, 'rec3':0, 'rec4':0}
        #self.norm_tag='nuc'
        self.norm_tag=norm_tag
        self.options=options
        self.margin=margin
        
        self.compute_dist=compute_dist(beta=self.beta,Ns=self.Ns,norm_tag=self.norm_tag,margin=self.margin,la=self.la,options=self.options)
        

        print ('\n###liwei0519 beta = {}, Ns = {}, norm_tag = {} margin = {} self.la={} self.options = {}###\n'.format(self.beta,self.Ns,self.norm_tag,self.margin,self.la, self.options))
    
    @staticmethod
    def add_args(parser):
        #===================================================================
        parser.add_argument('--margin', default=0.5, type=float, metavar='D',
                            help='margin for dist constraint')

        parser.add_argument('--beta', default=0.2, type=float, metavar='D',
                            help='beta for dist constraint')
                            
        parser.add_argument('--la', default=0.5, type=float, metavar='D',
                            help='lambda for dist constraint')

        parser.add_argument('--Ns', default=20, type=int, metavar='N',
                            help='beta for dist constraint')

        parser.add_argument('--dist-opt', type=str, default=None, metavar='STR',
                            help='different dist constraint: cos dnorm')
                            
        parser.add_argument('--norm-tag', type=str, default='fro', metavar='STR',
                            help='norm-tag')
        #======================================================================

    
    def forward_dist(self, model, sample):
        #assert (0==1)
        
        #net_output,Dist_M_sum,etc_dic = model(**sample['net_input'],tgt_tokens=sample['target'],tgt_lengths=sample['tgt_lengths'],compute_mod='compute_dist')
        
        #print ('net_output = {}'.format(net_output))
        #src_embeddings=net_output[-1]['sentemb']
        
        
        #print ('\n###########\nsample ={}\n##########\n'.format(sample))
        
        tgt_tokens=sample['target']
        tgt_lengths=sample['tgt_lengths']
        
        net_output = model(**sample['net_input'])
        src_sentemb=net_output[-1]['sentemb']
        
        tgt_sentemb=model.encoder(src_tokens=tgt_tokens, src_lengths=tgt_lengths)['sentemb']
        Dist_M_sum,etc_dic=self.compute_dist(src_sentemb, tgt_sentemb)
        
        #===========================================================================
        #options={}
        #options['dist_mode'] = 0
        
        return net_output,Dist_M_sum,etc_dic
        
    def forward(self, model, sample, reduce=True):
        '''
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        '''
        
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
     
        net_output,Dist_M_sum,etc_dic=self.forward_dist(model,sample)
        #print ('rev tgt_lengths = {}'.format(tgt_lengths))

        #embeddings = model.encoder(src_tokens, src_lengths)['sentemb']
        #print('src_embeddings = {}, tgt_embeddings = {}'.format(src_embeddings.size(),tgt_embeddings.size()))

        ori_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        #print ('loss = {} {}'.format(loss.size(),loss))
        
        
        #Dist_M_sum=dist_M.sum()

        #print ('loss = {}, Dist_M_sum = {}'.format(loss, Dist_M_sum))
        
        loss=0.5*ori_loss+self.beta*Dist_M_sum
        #print ('.5*loss+beta*Dist_M_sum = {} {}'.format(loss.size(),loss))
        
        

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            #'ori_loss': ori_loss,
            #'Dist_M_sum': Dist_M_sum,
            #'vec_en2de_mean_norm': etc_dic['vec_en2de_mean_norm'],
            #'vec_de2en_mean_norm': etc_dic['vec_de2en_mean_norm'],
        }
        
        
        '''
        end.record()
        torch.cuda.synchronize()
        self.timers['rec0']+=1
        self.timers['rec2']+=start.elapsed_time(end)
        
        print('pow syn {},start.elapsed_time(end) = {},t1 {}, t2 {}'.format(self.timers['rec0'],start.elapsed_time(end),self.timers['rec1'],self.timers['rec2']))
        '''
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        #batch_size=len([log.get('ori_loss', 0) for log in logging_outputs])
        #print (str(logging_outputs))
        
        #===================================================================================================
        #ori_loss=utils.item(sum(log.get('ori_loss', 0) for log in logging_outputs))
        #Dist_M_sum=utils.item(sum(log.get('Dist_M_sum', 0).float() for log in logging_outputs))
        #vec_en2de_mean_norm=utils.item(sum(log.get('vec_en2de_mean_norm', 0).float() for log in logging_outputs))
        #vec_de2en_mean_norm=utils.item(sum(log.get('vec_de2en_mean_norm', 0).float() for log in logging_outputs))
        #=======================================================================================================
        
        #print ('batch_size = {}, sample_size = {}, ori_loss = {}, Dist_M_sum = {} vec_en2de_mean_norm = {}, vec_de2en_mean_norm = {}'.format(batch_size, sample_size,ori_loss,Dist_M_sum,vec_en2de_mean_norm,vec_de2en_mean_norm))

        #print ('ori_loss= {} Dist_M_sum ={} vec_en2de_mean_norm = {}, vec_de2en_mean_norm ={}'.format(ori_loss,Dist_M_sum,vec_en2de_mean_norm,vec_de2en_mean_norm))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        
        #===========================================================================================
        #metrics.log_scalar('ori_loss', loss_sum , sample_size, round=3)
        #metrics.log_scalar('Dist_M_sum', Dist_M_sum , sample_size, round=3)
        #metrics.log_scalar('vec_en2de_mean_norm', vec_en2de_mean_norm , sample_size, round=3)
        #metrics.log_scalar('vec_de2en_mean_norm', vec_de2en_mean_norm , sample_size, round=3)
        
        #===========================================================================================
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
