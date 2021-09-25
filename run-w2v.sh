#!/bin/bash

echo "----------------------------------"
echo "📝 Preparing..."
echo "🧪 exp1 : CBOW | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 10 -t 0 -w 5 -msl 256
echo "🧪 exp2 : CBOW | window : 5 | learning rate : 0.005 "
python main.py -w2v -n exp2 -lr 0.005 -bs 64 -e 10 -t 0 -w 5 -msl 256
echo "🧪 exp3 : CBOW | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp3 -lr 0.001 -bs 64 -e 10 -t 0 -w 3 -msl 256
echo "🧪 exp4 : CBOW | window : 3 | learning rate : 0.005 "
python main.py -w2v -n exp4 -lr 0.005 -bs 64 -e 10 -t 0 -w 3 -msl 256
echo "🧪 exp5 : SG | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp5 -lr 0.001 -bs 64 -e 10 -t 1 -w 5 -msl 256
echo "🧪 exp6 : SG | window : 5 | learning rate : 0.005 "
python main.py -w2v -n exp6 -lr 0.005 -bs 64 -e 10 -t 1 -w 5 -msl 256
echo "🧪 exp7 : SG | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp7 -lr 0.001 -bs 64 -e 10 -t 1 -w 3 -msl 256
echo "🧪 exp8 : SG | window : 3 | learning rate : 0.005 "
python main.py -w2v -n exp8 -lr 0.005 -bs 64 -e 10 -t 1 -w 3 -msl 256
echo "✅ Training complete. Happy learning ♥️."