#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/fairseq-py/blob/master/data/prepare-iwslt14.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=10000

GZ=tr-en.tgz
GZ1=en-tr.txt.zip

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=tr
tgt=en
lang=$src-$tgt
prep=iwslt14.tokenized.$lang
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

# echo "Downloading data from ${URL}..."
cd $orig
# wget "$URL"

if [ -f $GZ ]; then
    echo "Original data available."
else
    echo "Data missing. Please add tar file in $orig folder"
    exit
fi

tar zxvf $GZ
unzip $GZ1 -d $lang
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    f1=SETIMES2.en-tr.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f $orig/$lang/$f1 | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
# Clean by ration 9 and remove sentences of length more than 175
perl $CLEAN -ratio 9 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$src-$tgt.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$src-$tgt.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.$src-$tgt.$l \
        $tmp/IWSLT14.TED.tst2010.$src-$tgt.$l \
        $tmp/IWSLT14.TED.tst2011.$src-$tgt.$l \
        $tmp/IWSLT14.TED.tst2012.$src-$tgt.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.$lang
BPE_CODE=$prep/bpe_code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/train.bpe.$L
    done
    for f in valid.$L ; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/valid.bpe.$L
    done
    for f in test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/test.bpe.$L
    done
done

for l in $src $tgt; do
    mv $tmp/train.$l $prep/
    mv $tmp/valid.$l $prep/
    mv $tmp/test.$l $prep/
done
