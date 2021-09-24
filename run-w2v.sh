#!/bin/bash

echo "----------------------------------"
echo "📝 Preparing..."
echo "🧪 exp1 : CBOW | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp3 : CBOW | window : 5 | learning rate : 0.0001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp4 : CBOW | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp6 : CBOW | window : 3 | learning rate : 0.0001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp10 : SG | window : 5 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp12 : SG | window : 5 | learning rate : 0.0001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp13 : SG | window : 3 | learning rate : 0.001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp15 : SG | window : 3 | learning rate : 0.0001 "
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp1 : CBOW | window : 5 | learning rate : 0.001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp3 : CBOW | window : 5 | learning rate : 0.0001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp4 : CBOW | window : 3 | learning rate : 0.001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp6 : CBOW | window : 3 | learning rate : 0.0001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp10 : SG | window : 5 | learning rate : 0.001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp12 : SG | window : 5 | learning rate : 0.0001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp13 : SG | window : 3 | learning rate : 0.001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "🧪 exp15 : SG | window : 3 | learning rate : 0.0001 w/ Fasttext"
python main.py -w2v -n exp1 -lr 0.001 -bs 64 -e 5 -t 0 -w 5 -msl 128
echo "✅ Training complete. Happy learning ♥️."