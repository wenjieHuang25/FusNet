#!/bin/bash

usage()
{
  echo "$(basename "$0") [-h] INTERS DNASE TFPEAKS NAME OUTDIR"
  echo "-- Generate positve and negative loop and anchor files."

  echo "where:"
  echo "-h           Display this help message"
  echo "INTERS       The path of interaction file"
  echo "DNASE        The path of DNase-seq file"
  echo "TFPEAKS      The path of ChIP-seq file"
  echo "NAME         The name or the sample"
  echo "OUTDIR       The path of output directory"
}


if [ "$1" != "" ]; then
    case $1 in
        -h | --help )           usage
                                exit
                                ;;
    esac
fi

if [ $# -lt 5 ]; then
  usage
  exit
fi

INTERS=$1
DNASE=$2
TFPEAKS=$3
NAME=$4
OUTDIR=$5

DIR=$(dirname "$0")

echo "Removing interactions whose two anchors are overlapping or on different chromosomes"
cat $INTERS | awk '$1==$4 && ($3<$5 || $6<$2)' > ${OUTDIR}/${NAME}_positive_loops.bedpe

positive_anchors=${OUTDIR}/${NAME}_positive_anchors.bed

cat ${OUTDIR}/${NAME}_positive_loops.bedpe \
    | awk 'BEGIN{FS=OFS="\t"}{printf("%s\t%s\t%s\n%s\t%s\t%s\n", $1, $2, $3, $4, $5, $6)}' \
    | sort -u > ${positive_anchors}

echo "Generating random anchor pairs"
python preprocess/generate_random_anchor_pairs.py $NAME $OUTDIR

echo "Generating random TF peak pairs"
python preprocess/generate_random_pairs_bed.py $NAME $TFPEAKS tf ${OUTDIR}

echo "Generating random DNase pairs"
python preprocess/generate_random_pairs_bed.py $NAME $DNASE dnase ${OUTDIR}

echo "Filtering TF peak pairs"
pairToPair -a ${OUTDIR}/${NAME}.random_tf_peak_pairs.bedpe -b $INTERS -type notboth  \
    | pairToPair -a stdin -b  ${OUTDIR}/${NAME}.no_intra_all.negative_pairs.bedpe   -type notboth \
    | pairToPair -a stdin -b  ${OUTDIR}/${NAME}.only_intra_all.negative_pairs.bedpe  -type notboth \
    | uniq > ${OUTDIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe

echo "Filtering DNase pairs"
pairToPair -a ${OUTDIR}/${NAME}.shuffled_neg_anchor.neg_pairs.bedpe -b $INTERS -type notboth \
    | pairToPair -a stdin -b  ${OUTDIR}/${NAME}.no_intra_all.negative_pairs.bedpe -type notboth \
    | pairToPair -a stdin -b  ${OUTDIR}/${NAME}.only_intra_all.negative_pairs.bedpe -type notboth \
    | uniq \
    | pairToPair -a stdin -b ${OUTDIR}/${NAME}.random_tf_peak_pairs.filtered.bedpe -type notboth \
    | uniq > ${OUTDIR}/${NAME}.shuffled_neg_anchor.neg_pairs.filtered.tf_filtered.bedpe

echo "Sampling 5x negative samples"
python preprocess/generate_5fold_neg.py $NAME ${OUTDIR}

echo "Generate train/test/valid data"
python preprocess/generate_data.py -m 1000 -e 500 \
--pos_files ${OUTDIR}/${NAME}_positive_loops.bedpe \
--neg_files ${OUTDIR}/${NAME}_negative_loops.bedpe \
-g data/hg19.fa -n ${NAME}_loops -o ${OUTDIR}