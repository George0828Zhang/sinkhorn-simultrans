#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh
DATA_ROOT=/media/george/Data/wmt15
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
SCRIPTS=~/utility/mosesdecoder/scripts
# source ~/envs/apex/bin/activate

SRC=de
TGT=en
vocab=32000
vtype=unigram
workers=4

# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

DATA=${DATA_ROOT}/${SRC}-${TGT}

URLS=(
    "https://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://www.statmt.org/wmt15/training-parallel-nc-v10.tgz"
    "https://www.statmt.org/wmt15/dev-v2.tgz"
    "https://www.statmt.org/wmt15/test.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v10.tgz"
    "dev-v2.tgz"
    "test.tgz"
)
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "news-commentary-v10.de-en"
)

orig=${DATA}/orig
prep=${DATA}/prep
ready=${DATA}/ready
bin=${DATA}/data-bin
mkdir -p $orig $prep $ready $bin

echo "downloading data"
cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        tar zxvf $file
    fi
done
cd ..

echo "pre-processing train data..."
for l in ${SRC} ${TGT}; do
    rm -f $prep/train.dirty.$l
    for f in "${CORPORA[@]}"; do
        echo "precprocess train $f.$l"
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $LC >> $prep/train.dirty.$l
    done
done

echo "pre-processing valid data..."
for l in ${SRC} ${TGT}; do
    if [ "$l" == "${SRC}" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2013-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/valid.dirty.$l
done

echo "pre-processing test data..."
for l in ${SRC} ${TGT}; do
    if [ "$l" == "${SRC}" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newstest2015-${SRC}${TGT}-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $LC > $prep/test.$l
done

# filter empty pairs
perl $CLEAN -ratio 1000 $prep/train.dirty ${SRC} ${TGT} $prep/train 1 10000
perl $CLEAN -ratio 1000 $prep/valid.dirty ${SRC} ${TGT} $prep/valid 1 10000

# SPM
SPM_PREFIX=$prep/spm_${vtype}${vocab}
for l in ${SRC} ${TGT}; do
    SPM_MODEL=${SPM_PREFIX}_${l}.model
    DICT=${SPM_PREFIX}_${l}.txt
    BPE_TRAIN=$prep/bpe-train.$l

    if [[ ! -f $SPM_MODEL ]]; then
        if [ -f $BPE_TRAIN ]; then
            echo "$BPE_TRAIN found, skipping concat."
        else
            train=$prep/train.$l
            default=1000000
            total=$(cat $train | wc -l)
            echo "lang $l total: $total."
            if [ "$total" -gt "$default" ]; then
                cat $train | \
                shuf -r -n $default >> $BPE_TRAIN
            else
                cat $train >> $BPE_TRAIN
            fi     
        fi               

        echo "spm_train on $BPE_TRAIN..."
        ccvg=1.0
        if [[ ${l} == "zh" ]]; then
            ccvg=0.9995
        fi
        python $spm_train --input=$BPE_TRAIN \
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
    for split in train valid test; do
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

# filter ratio and maxlen < 1024
perl $CLEAN -ratio 9 $ready/train ${SRC} ${TGT} $ready/train.clean 1 1024

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/train.clean \
    --validpref ${ready}/valid \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --srcdict ${SPM_PREFIX}_${SRC}.txt \
    --tgtdict ${SPM_PREFIX}_${TGT}.txt