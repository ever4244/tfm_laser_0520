#!/bin/bash

#bpe=europarl_en_de_es_fr/bpe.40k
#data_bin=data-bin/europarl.de_en_es_fr.bpe40k

bpe=europarl_en_de_es_fr_it/bpe.50k
data_bin=data-bin/europarl.de_en_es_fr_it.bpe50k
# for lang_pair in de-en de-es de-fr en-es en-fr es-fr en-it es-it fr-it de-it; do
#     src=`echo $lang_pair | cut -d'-' -f1`
#     tgt=`echo $lang_pair | cut -d'-' -f2`
#     rm $data_bin/dict.$src.txt $data_bin/dict.$tgt.txt
#     fairseq-preprocess --source-lang $src --target-lang $tgt \
#         --trainpref $bpe/train.$src-$tgt \
#         --joined-dictionary --tgtdict $bpe/vocab \
#         --destdir $data_bin \
#         --workers 40
# done



#for lang_pair in de-en de-es de-fr en-es en-fr es-fr; do
#    src=`echo $lang_pair | cut -d'-' -f1`
#    tgt=`echo $lang_pair | cut -d'-' -f2`
#    rm $data_bin/dict.$src.txt $data_bin/dict.$tgt.txt
#    fairseq-preprocess --source-lang $src --target-lang $tgt \
#        --trainpref $bpe/train.$src-$tgt \
#        --joined-dictionary --tgtdict $bpe/vocab \
#        --destdir $data_bin \
#        --workers 20
#done

#


decoder_layers=1

sentemb_mod='first'
#sentemb_mod='keymaster04'
#sentemb_mod='keymaster_nw'


#margin=0.25
margin=0.5
la=0.5
#beta=0.5
beta=0.25
Ns=20
#dist_opt='cos'
dist_opt='dnorm+'
#tag='_8kx16_5lan_shremb'

#arch=transformer_wmt_en_de_bigmid_t2t_lw
arch='transformer_iwslt_de_en_lw_mod1024X4096X16dec1'
#arch=transformer_vaswani_wmt_en_de_big_norm_lw
#arch=transformer_wmt_en_de_big_t2t_lw
#arch=transformer_wmt_en_de_big_t2t_lw_dec3
arch_id=${arch}_dec${decoder_layers}

#64000
max_tokens=4000
update_freq=16
max_pos=2000

#max_tokens=2000
#update_freq=16
#max_pos=2000

#max_tokens=4000
#update_freq=8
#max_pos=2000

#pre_model="_pre_b025m5ep10"
#pre_model=""
#=================================
#
tag1='_5lan_fp16_normbef_mp'${max_pos}
tag2='_initB_noshr'$pre_model
#tag2='_initB_shremb'
#==================================
tag=${tag1}${tag2}

criterion=label_smoothed_cross_entropy_dist

if [[ $criterion == *"_dist"* ]]; then
    checkpoint=checkpoints_bak/128k_${max_tokens}x${update_freq}_emb${sentemb_mod}_m${margin}_la${la}_b${beta}_N${Ns}_${dist_opt}${tag}_${arch_id}
    liwei_crt="--margin ${margin} --beta ${beta} --la ${la} --Ns ${Ns}  --dist-opt ${dist_opt} "
else
    checkpoint=checkpoints_bak/128k_${max_tokens}x${update_freq}_emb${sentemb_mod}_nodist${tag}_${arch_id}
    liwei_crt=""
fi

#=========================

mkdir -p checkpoints_bak
mkdir -p $checkpoint


#=========================


if [[ $tag2 == *"_shremb"* ]]; then
    liwei_opt="--share-all-embeddings"
else
    liwei_opt=""
fi


lanpair="de-en,de-es,en-es,es-en,fr-en,fr-es,it-en,it-es"
#lanpair="en-es,es-en"



#lanpair="de-en,de-es,en-es,es-en,fr-en,fr-es"

#de-en,de-es,en-es,es-en,fr-en,fr-es,it-en,it-es
#de-en,de-es,en-es,es-en,fr-en,fr-es

#label_smoothed_cross_entropy_dist
#transformer_iwslt_de_en_lw_mod1024X4096X16dec1
#transformer_iwslt_de_en_lw_mod

#laser_lw_mod_tfm_0701

export  MKL_THREADING_LAYER=GNU

export CUDA_VISIBLE_DEVICES="0,1"
fairseq-train $data_bin \
    --ddp-backend=no_c10d \
    --no-progress-bar \
    --max-epoch 50 \
    --task translation_laser --arch $arch \
    --lang-pairs $lanpair \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
    --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --label-smoothing 0.1 --criterion $criterion \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir $checkpoint \
    --max-tokens $max_tokens \
    --update-freq $update_freq --valid-subset train --disable-validation \
    --user-dir tfm_laser_0520/ --fp16 \
    --max-source-positions $max_pos --max-target-positions $max_pos --num-workers 32 --encoder-normalize-before --decoder-normalize-before \
    --left-pad-source False \
    --sentemb-mod $sentemb_mod  --data-buffer-size 8 --init-mod 'BERT' ${liwei_opt} ${liwei_crt} --decoder-layers ${decoder_layers}
    

#echo 'end'    
    
    
    
    
    
    
    
    
    
    #--margin $margin --beta $beta --la $la --Ns $Ns  --dist-opt $dist_opt 
    #--reset-optimizer
    

#--reset-optimizer
#append_bos
#--append-bos
#--ddp-backend=no_c10d \
#--share-encoders
#--share-decoders --share-decoder-input-output-embed 

#--share-decoders --share-decoder-input-output-embed \
#multilingual_translation_lw
#translation_laser
#checkpoint=checkpoints/multilingual_transformer_sample
#mkdir -p $checkpoint
#fairseq-train $data_bin \
#  --max-epoch 20 \
#  --ddp-backend=no_c10d \
#  --task multilingual_translation  \
#  --lang-pairs de-en,de-es,en-es,es-en,fr-en,fr-es \
#  --optimizer adam --adam-betas '(0.9, 0.98)' \
#  --lr 0.001 --criterion cross_entropy_dist \
#  --dropout 0.1 --save-dir $checkpoint \
#  --max-tokens 2000 \
#  --valid-subset train --disable-validation \
#  --no-progress-bar --log-interval 1000 \
#  --user-dir laser_lw_mod/  --update-freq 10 \
#  --arch multilingual_transformer_iwslt_de_en_lw  --share-decoder-input-output-embed


#--task translation_laser  \