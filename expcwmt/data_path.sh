export SRC=en
export TGT=zh
export DATA=/livingrooms/george/cwmt/zh-en/data-bin
export WANDB_START_METHOD=thread
export FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate
