#!/usr/bin/env bash

ALGO="DartFaceNet256-sx2-KD2"
NUM_ITERATIONS="295672backbone.pth"
EPOCH="26"
DATA="/data/fboutros/megaface_testpack_v1.0"
MODELSDIR="//home/fboutros/DartFaceNet/outputKD"

ROOT="/home/fboutros/DartFaceNet"

FEATUREOUT="$ROOT/output_features/$ALGO-$EPOCH"
FEATUREOUTCLEAN="$ROOT/output_features_clean/$ALGO-$EPOCH"


CUDA_VISIBLE_DEVICES=3 python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model "$MODELSDIR/$ALGO/$NUM_ITERATIONS" --megaface-data "$DATA" --output "$FEATUREOUT"
python -u remove_noises.py --algo "$ALGO" --megaface-data "$DATA" --feature-dir-input "$FEATUREOUT" --feature-dir-out "$FEATUREOUTCLEAN"

DEVKIT="/home/psiebke/Thesis/dartFaceNetKD/devkit/experiments/"

cd "$DEVKIT"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/maklemt/anaconda3/envs/envMegaface/lib/"
python -u run_experiment.py "$FEATUREOUTCLEAN/megaface" "$FEATUREOUTCLEAN/facescrub" _"$ALGO".bin "$ROOT/$ALGO-$EPOCH/" -s 1000000 -p ../templatelists/facescrub_features_list.json

python -u run_experiment.py "$FEATUREOUT/megaface" "$FEATUREOUT/facescrub" _"$ALGO".bin "$ROOT/$ALGO-$EPOCH-noisy/" -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

