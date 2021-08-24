#!/usr/bin/env bash
. ~/envs/apex/bin/activate
MODEL=bert-base-multilingual-cased
SRC=en
TGT=zh
REORDER=../scripts/reorder.py
PREFIX=/livingrooms/george/cwmt/zh-en/ready/distill_${TGT}
OUTDIR=./alignments.results

SRCTOK=$(mktemp)
TGTTOK=$(mktemp)
CORPUS=$(mktemp)

# echo "extracting text ..."
# sed 's/▁//g' ${PREFIX}.${SRC} > ${SRCTOK}
# sed 's/▁//g' ${PREFIX}.${TGT} > ${TGTTOK}

echo "extracting text ..."
cat ${PREFIX}.${SRC} > ${SRCTOK}
cat ${PREFIX}.${TGT} > ${TGTTOK}

echo "aligning ..."
mkdir -p ${OUTDIR}
ALIGNOUT=${OUTDIR}/$(basename ${PREFIX}).${SRC}-${TGT}
if [ -f "${ALIGNOUT}" ]; then
	echo "${ALIGNOUT} exists, skipping alignment"
else
	paste ${SRCTOK} ${TGTTOK} | sed "s/\t/ ||| /" > ${CORPUS}
	python -m awesome_align.run_align \
    --output_file=${ALIGNOUT} \
    --model_name_or_path=${MODEL} \
    --data_file=${CORPUS} \
    --extraction 'softmax' \
    --batch_size 128
fi

echo "reordering ..."
REORDEROUT=${OUTDIR}/$(basename ${PREFIX}).${TGT}.reord
if [ -f "${REORDEROUT}" ]; then
	echo "${REORDEROUT} exists, skipping reordering"
else	
	python ${REORDER} \
        -s ${PREFIX}.${SRC} \
        -t ${PREFIX}.${TGT} \
        -a ${ALIGNOUT} -o ${REORDEROUT}
fi

rm -f $SRCTOK
rm -f $TGTTOK
rm -f $CORPUS