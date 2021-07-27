#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/translation/prepare-iwslt14.sh
DATA_ROOT=/media/george/Data/um-corpus
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
SCRIPTS=~/utility/mosesdecoder/scripts
# source ~/envs/apex/bin/activate

SRC=zh
TGT=en
lang=zh-en
vocab=8000
vtype=unigram
workers=4

# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
# NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

DATA=${DATA_ROOT}/${SRC}-${TGT}

orig=${DATA}/orig
prep=${DATA}/prep
tmp=$prep/tmp
ready=${DATA}/ready
bin=${DATA}/data-bin
mkdir -p $orig $prep $tmp $ready $bin


rm -f $orig/train.${SRC} $orig/train.${TGT}
rm -f $orig/valid.${SRC} $orig/valid.${TGT}
for f in `ls ${DATA_ROOT}/UM-Corpus/data/Bilingual/**/*.txt`; do
    sed -n 'n;p' $f >> $orig/train.zh
    sed -n 'p;n' $f >> $orig/train.en
done
test_f=${DATA_ROOT}/UM-Corpus/data/Testing/Testing-Data.txt
grep -vE "(\([0-9]+\).*test data|（[0-9]+）.*测试集)" $test_f | \
    sed -n 'n;p' > $orig/valid.zh
grep -vE "(\([0-9]+\).*test data|（[0-9]+）.*测试集)" $test_f | \
    sed -n 'p;n' > $orig/valid.en

echo "pre-processing train data..."
for l in ${SRC} ${TGT}; do
    cat $orig/train.$l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/train.dirty.$l
done
echo "pre-processing valid data..."
for l in ${SRC} ${TGT}; do
    cat $orig/valid.$l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/valid.$l
done

perl $CLEAN -ratio 1000 $prep/train.dirty ${SRC} ${TGT} $prep/train 1 1000

# SPM
SPM_PREFIX=$prep/spm_${vtype}${vocab}
for l in ${SRC} ${TGT}; do
    SPM_MODEL=${SPM_PREFIX}_${l}.model
    DICT=${SPM_PREFIX}_${l}.txt

    if [[ ! -f $SPM_MODEL ]]; then
        echo "spm_train on $prep/train.$l ..."
        ccvg=1.0
        if [[ ${l} == "zh" ]]; then
            ccvg=0.9995
        fi
        python $spm_train --input=$prep/train.$l \
            --model_prefix=${SPM_PREFIX}_${l} \
            --vocab_size=$vocab \
            --character_coverage=$ccvg \
            --model_type=$vtype \
            --normalization_rule_name=nmt_nfkc_cf
        
        cut -f1 ${SPM_PREFIX}_${l}.vocab | tail -n +4 | sed "s/$/ 100/g" > $DICT
        cp $SPM_MODEL $bin/$(basename $SPM_MODEL)
        cp $DICT $bin/$(basename $DICT)
    fi

    echo "Using SPM model $SPM_MODEL"
    for split in train valid; do
        if [ -f $ready/$split.$l ]; then
            echo "found $ready/$split.$l, skipping spm_encode"
        else
            echo "spm_encode to $split.$l..."
            python $spm_encode --model=$SPM_MODEL \
                --output_format=piece \
                < $prep/$split.$l > $ready/$split.$l
        fi
    done
done

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/valid \
    --destdir ${bin} \
    --workers ${workers} \
    --srcdict ${SPM_PREFIX}_${SRC}.txt \
    --tgtdict ${SPM_PREFIX}_${TGT}.txt