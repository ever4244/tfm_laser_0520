# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
#import cross_entropy_dist
from .cross_entropy_dist import CrossEntropyCriterion_dist

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_dist')
class LabelSmoothedCrossEntropyCriterion_dist(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, beta=0.2,Ns=20,margin=0.25,dist_opt='cos'):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.beta=beta
        self.Ns=Ns
        self.margin=margin
        self.timers={'rec0':0, 'rec1':0, 'rec2':0, 'rec3':0, 'rec4':0}
        #self.norm_tag='nuc'
        self.norm_tag='fro'

        self.options=dist_opt
        #self.options=None

        print ('\n### LabelSmoothedCE beta = {}, Ns = {}, norm_tag = {} margin = {}###\n'.format(self.beta,self.Ns,self.norm_tag,self.margin))
        self.ce_dist=CrossEntropyCriterion_dist(task,sentence_avg,beta=self.beta,Ns=self.Ns,norm_tag=self.norm_tag,margin=self.margin,options=self.options)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

        
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


        # fmt: on
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        net_output,Dist_M_sum,etc_dic=self.ce_dist.forward_dist(model,sample)
        
        ori_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss=0.5*ori_loss+self.beta*Dist_M_sum
        
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            #'ori_loss': ori_loss,
            #'Dist_M_sum': Dist_M_sum,
            #'vec_en2de_mean_norm': etc_dic['vec_en2de_mean_norm'],
            #'vec_de2en_mean_norm': etc_dic['vec_de2en_mean_norm'],
        
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        #ori_loss=utils.item(sum(log.get('ori_loss', 0) for log in logging_outputs))
        #Dist_M_sum=utils.item(sum(log.get('Dist_M_sum', 0).float() for log in logging_outputs))
        #vec_en2de_mean_norm=utils.item(sum(log.get('vec_en2de_mean_norm', 0).float() for log in logging_outputs))
        #vec_de2en_mean_norm=utils.item(sum(log.get('vec_de2en_mean_norm', 0).float() for log in logging_outputs))
        
        
        #metrics.log_scalar('ori_loss', loss_sum , sample_size, round=3)
        #metrics.log_scalar('Dist_M_sum', Dist_M_sum , sample_size, round=3)
        #metrics.log_scalar('vec_en2de_mean_norm', vec_en2de_mean_norm , sample_size, round=3)
        #metrics.log_scalar('vec_de2en_mean_norm', vec_de2en_mean_norm , sample_size, round=3)
        
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
