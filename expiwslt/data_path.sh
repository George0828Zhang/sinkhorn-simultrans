export SRC=de
export TGT=en
export DATA=/media/george/Data/iwslt14/de-en/data-bin

export FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# . ~/envs/apex/bin/activate
