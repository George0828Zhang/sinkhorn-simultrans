source ~/envs/sacrebleu2/bin/activate
WORKERS=2
TGT=zh
REF=(
    "/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${TGT}.1"
    "/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${TGT}.2"
    "/livingrooms/george/cwmt/zh-en/prep/test.en-zh.${TGT}.3"
)
BASELINE=wait_1_enzh_distill.cwmt/prediction
SYSTEMS=(
    "teacher_cwmt_enzh.cwmt/prediction"
    "ctc_delay1.cwmt/prediction"
)
OUTPUT=cwmt.${TGT}.systems.result
python -m sacrebleu ${REF[@]} -i ${BASELINE} ${SYSTEMS[@]} \
    --paired-jobs ${WORKERS} \
    -m bleu ter \
    --tok zh -lc \
    --ter-asian-support --ter-normalize \
    --paired-bs | tee ${OUTPUT}