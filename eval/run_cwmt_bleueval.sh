source ~/utility/sacrebleu/sacrebleu2/bin/activate
SRC=en
TGT=zh
DIR=cwmt_${TGT}-results
WORKERS=2
REF=(
    "/media/george/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.1"
    "/media/george/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.2"
    "/media/george/Data/cwmt/zh-en/prep/test.${SRC}-${TGT}.${TGT}.3"
)

# Normal
for DELAY in 1 3 5 7 9; do
    BASELINE="${DIR}/wait_${DELAY}_${SRC}${TGT}_distill.cwmt/prediction"
    SYSTEMS=(
        "${DIR}/ctc_delay${DELAY}.cwmt/prediction"
        "${DIR}/sinkhorn_delay${DELAY}.cwmt/prediction"
    )

    OUTPUT=${DIR}/quality-results.cwmt/delay${DELAY}-systems
    mkdir -p $(dirname ${OUTPUT})
    python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
        --paired-jobs ${WORKERS} \
        -m bleu \
        --width 2 \
        --tok zh -lc \
        --paired-bs | tee ${OUTPUT}
done

# Full-sentence
TEACHER="${DIR}/teacher_cwmt_${SRC}${TGT}.cwmt/prediction"
OUTPUT=${DIR}/quality-results.cwmt/full_sentence-systems
mkdir -p $(dirname ${OUTPUT})
python -m sacrebleu ${REF[@]} -i ${BASELINE} \
    --paired-jobs ${WORKERS} \
    -m bleu \
    --width 2 \
    --tok zh -lc \
    --confidence | tee ${OUTPUT}