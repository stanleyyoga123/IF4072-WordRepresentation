#!/bin/bash

echo "----------------------------------"
echo "📝 Preparing Test run..."
echo "🧪 exp1 : CBOW | window : 5 | learning rate : 0.001 "
python main.py -w2v -n example -lr 0.001 -bs 128 -e 5 -t 1 -w 5 -msl 128
echo "✅ Training complete. Happy learning ♥️."