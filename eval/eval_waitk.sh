#!/usr/bin/env bash
TASK=wait_1_deen_distill
SPLIT=valid
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 1 --max-len-a 1.2 --max-len-b 10 --remove-bpe sentencepiece"

python -m fairseq_cli.generate ${DATA} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation_infer \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 8000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None, "load_pretrained_decoder_from": None}' \
  ${GENARGS} ${EXTRAARGS}