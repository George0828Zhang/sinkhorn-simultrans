source ~/utility/sacrebleu/sacrebleu2/bin/activate
SRC=de
TGT=en
DIR=wmt15_${TGT}-results
WORKERS=2
REF=(
    "/media/george/Data/wmt15/de-en/prep/test.${TGT}"
)

# Normal
for DELAY in 1 3 5 7 9; do
    BASELINE="${DIR}/wait_${DELAY}_${SRC}${TGT}_distill.wmt15/prediction"
    SYSTEMS=(
        "${DIR}/ctc_delay${DELAY}.wmt15/prediction"
        "${DIR}/sinkhorn_delay${DELAY}.wmt15/prediction"
    )

    OUTPUT=${DIR}/quality-results.wmt15/delay${DELAY}-systems
    mkdir -p $(dirname ${OUTPUT})
    python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
        --paired-jobs ${WORKERS} \
        -m bleu \
        --width 2 \
        --tok 13a -lc \
        --paired-bs | tee ${OUTPUT}
done

# Full-sentence
TEACHER="${DIR}/teacher_wmt15_${SRC}${TGT}.wmt15/prediction"
OUTPUT=${DIR}/quality-results.wmt15/full_sentence-system
mkdir -p $(dirname ${OUTPUT})
python -m sacrebleu ${REF[@]} -i ${BASELINE} \
    --paired-jobs ${WORKERS} \
    -m bleu \
    --width 2 \
    --tok 13a -lc \
    --confidence | tee ${OUTPUT}