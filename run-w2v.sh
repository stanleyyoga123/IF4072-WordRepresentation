#!/bin/bash

echo "----------------------------------"
echo "📝 Preparing..."
echo "🧪 exp1 : CBOW | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp2 : CBOW | window : 5 | learning rate : 0.0001 "
python main.py -w2v -n exp2 -lr 0.0001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp3 : CBOW | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp3 -lr 0.001 -bs 64 -e 5 -t 0 -w 3 -msl 128
echo "🧪 exp4 : CBOW | window : 3 | learning rate : 0.0001 "
python main.py -w2v -n exp4 -lr 0.0001 -bs 64 -e 5 -t 0 -w 3 -msl 128
echo "🧪 exp5 : SG | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp5 -lr 0.001 -bs 64 -e 5 -t 1 -w 5 -msl 128
echo "🧪 exp6 : SG | window : 5 | learning rate : 0.0001 "
python main.py -w2v -n exp6 -lr 0.0001 -bs 64 -e 5 -t 1 -w 5 -msl 128
echo "🧪 exp7 : SG | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp7 -lr 0.001 -bs 64 -e 5 -t 1 -w 3 -msl 128
echo "🧪 exp8 : SG | window : 3 | learning rate : 0.0001 "
python main.py -w2v -n exp8 -lr 0.0001 -bs 64 -e 5 -t 1 -w 3 -msl 128
echo "🧪 exp9 : CBOW | window : 5 | learning rate : 0.001 w/ Fasttext"
python main.py -ft -n exp9 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp10 : CBOW | window : 5 | learning rate : 0.0001 w/ Fasttext"
python main.py -ft -n exp10 -lr 0.0001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp11 : CBOW | window : 3 | learning rate : 0.001 w/ Fasttext"
python main.py -ft -n exp11 -lr 0.001 -bs 64 -e 5 -t 0 -w 3 -msl 128
echo "🧪 exp12 : CBOW | window : 3 | learning rate : 0.0001 w/ Fasttext"
python main.py -ft -n exp12 -lr 0.0001 -bs 64 -e 5 -t 0 -w 3 -msl 128
echo "🧪 exp13 : SG | window : 5 | learning rate : 0.001 w/ Fasttext"
python main.py -ft -n exp13 -lr 0.001 -bs 64 -e 5 -t 1 -w 5 -msl 128
echo "🧪 exp14 : SG | window : 5 | learning rate : 0.0001 w/ Fasttext"
python main.py -ft -n exp14 -lr 0.0001 -bs 64 -e 5 -t 1 -w 5 -msl 128
echo "🧪 exp15 : SG | window : 3 | learning rate : 0.001 w/ Fasttext"
python main.py -ft -n exp15 -lr 0.001 -bs 64 -e 3 -t 1 -w 5 -msl 128
echo "🧪 exp16 : SG | window : 3 | learning rate : 0.0001 w/ Fasttext"
python main.py -ft -n exp16 -lr 0.0001 -bs 64 -e 3 -t 1 -w 5 -msl 128
echo "✅ Training complete. Happy learning ♥️."