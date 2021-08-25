#!/usr/bin/env bash
DATA_ROOT=/livingrooms/george/wmt15
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
SCRIPTS=~/utility/mosesdecoder/scripts
source ~/envs/apex/bin/activate
DECODED=./monotonic.results/generate-train.txt

SRC=de
TGT=en
lang=de-en
vocab=32000
vtype=unigram
workers=4

TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

DATA=${DATA_ROOT}/${lang}
SPM_PREFIX=${DATA_ROOT}/${lang}/data-bin/spm_${vtype}${vocab}


prep=${DATA}/prep
ready=${DATA}/ready
bin=${DATA}/data-bin
newbin=${DATA}/data-bin/tmp2
mkdir -p $prep $ready $bin $newbin

echo "pre-processing monotonic data..."
grep -E "S-[0-9]+" ${DECODED} | cut -f2 > $prep/monotonic_${TGT}.dirty.${SRC}
grep -E "H-[0-9]+" ${DECODED} | cut -f3 > $prep/monotonic_${TGT}.dirty.${TGT}

# filter empty pairs
perl $CLEAN -ratio 1000 $prep/monotonic_${TGT}.dirty ${SRC} ${TGT} $prep/monotonic_${TGT} 1 10000

echo "Using SPM model $SPM_MODEL"
for l in ${SRC} ${TGT}; do
    SPM_MODEL=${SPM_PREFIX}_${l}.model

    f=monotonic_${TGT}.$l
    if [ -f $ready/$f ]; then
        echo "found $ready/$f, skipping spm_encode"
    else
        echo "spm_encode to ${f}..."
        python $spm_encode --model=$SPM_MODEL \
            --output_format=piece \
            < $prep/$f > $ready/$f
    fi
done

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/monotonic_${TGT} \
    --destdir ${newbin} \
    --workers ${workers} \
    --srcdict ${SPM_PREFIX}_${SRC}.txt \
    --tgtdict ${SPM_PREFIX}_${TGT}.txt

for l in ${SRC} ${TGT}; do
    for ext in bin idx; do
        cp ${newbin}/train.${SRC}-${TGT}.$l.$ext ${bin}/train_monotonic_${TGT}.$lang.$l.$ext
        ln -s ${bin}/train_distill_${TGT}.$lang.$l.$ext ${bin}/train_monotonic_${TGT}1.$lang.$l.$ext
    done
done
