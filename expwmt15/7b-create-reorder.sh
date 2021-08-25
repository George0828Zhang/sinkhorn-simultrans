#!/usr/bin/env bash
DATA_ROOT=/livingrooms/george/wmt15
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# SCRIPTS=~/utility/mosesdecoder/scripts
source ~/envs/apex/bin/activate

SRC=de
TGT=en
lang=de-en
vocab=32000
vtype=unigram
workers=4

REORDERED=./alignments.results/distill_${TGT}.${TGT}.reord

# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
# CLEAN=$SCRIPTS/training/clean-corpus-n.perl
# NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
# REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
# LC=$SCRIPTS/tokenizer/lowercase.perl

# spm_train=$FAIRSEQ/scripts/spm_train.py
# spm_encode=$FAIRSEQ/scripts/spm_encode.py

DATA=${DATA_ROOT}/${lang}
SPM_PREFIX=${DATA_ROOT}/${lang}/data-bin/spm_${vtype}${vocab}

ready=${DATA}/ready
bin=${DATA}/data-bin
newbin=${DATA}/data-bin/tmp3
mkdir -p $ready $bin $newbin

echo "pre-processing distill data..."
cat $ready/distill_${TGT}.${SRC} > $ready/reorder_${TGT}.${SRC}
cat ${REORDERED} > $ready/reorder_${TGT}.${TGT}

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/reorder_${TGT} \
    --destdir ${newbin} \
    --workers ${workers} \
    --srcdict ${SPM_PREFIX}_${SRC}.txt \
    --tgtdict ${SPM_PREFIX}_${TGT}.txt

for l in ${SRC} ${TGT}; do
    for ext in bin idx; do
        cp ${newbin}/train.${SRC}-${TGT}.$l.$ext ${bin}/train_reorder_${TGT}.$lang.$l.$ext
    done
done
