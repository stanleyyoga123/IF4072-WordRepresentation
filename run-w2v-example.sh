#!/bin/bash

echo "----------------------------------"
echo "๐ Preparing Test run..."
echo "๐งช exp1 : CBOW | window : 5 | learning rate : 0.0001 "
python main.py -ft -n example -lr 0.0001 -bs 128 -e 3 -t 1 -w 5 -msl 128
echo "โ Training complete. Happy learning โฅ๏ธ."