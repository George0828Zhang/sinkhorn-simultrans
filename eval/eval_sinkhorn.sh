#!/usr/bin/env bash
TASK=sinkhorn_delay1_eos
SPLIT=test
EXP=../expiwslt
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=checkpoint_best.pt
RESULT=${TASK}.$(basename $(dirname $(dirname ${DATA})))

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--remove-bpe sentencepiece"
# --from-encoder

python -m fairseq_cli.generate ${DATA} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation_infer \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 8000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None, "load_pretrained_decoder_from": None}' \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}