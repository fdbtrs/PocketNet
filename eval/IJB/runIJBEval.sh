#!/usr/bin/env bash

MODEL=DartFaceNet256-sx2-KD2
ITERS=295672backbone.pth
TARGET="IJBC"
OUTPUT="$MODEL-$TARGET-E25"
CUDA_VISIBLE_DEVICES=2 python eval_ijbc.py --model-prefix "/home/fboutros/DartFaceNet/outputKD/$MODEL/$ITERS" --image-path "/data/fboutros/IJB_release/IJB_release/$TARGET" --job $OUTPUT --target $TARGET
