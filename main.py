import argparse

from src.word_embedding import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Word 2 vec with context
    parser.add_argument(
        "-b", "--bert", action="store_true", required=False, help="Train BERT Model"
    )
    parser.add_argument(
        "-ib",
        "--indobert",
        action="store_true",
        required=False,
        help="Train IndoBERT Model",
    )

    # Word 2 vec with no context
    parser.add_argument(
        "-w2v",
        "--word2vec",
        action="store_true",
        required=False,
        help="Train Word2Vec Model",
    )
    parser.add_argument(
        "-ft",
        "--fasttext",
        action="store_true",
        required=False,
        help="Train Fasttext Model",
    )
    parser.add_argument("-w", "--window", required=False, help="Window size")
    parser.add_argument("-t", "--type", required=False, help="1: Skip-gram, else: CBOW")

    # All model
    parser.add_argument("-n", "--name", required=False, help="Model Name")
    parser.add_argument(
        "-msl", "--maxseqlen", required=False, help="Max Sequence Length"
    )
    parser.add_argument("-bs", "--batchsize", required=False, help="Batch Size Used")
    parser.add_argument(
        "-lr", "--learningrate", required=False, help="Learning rate Used"
    )
    parser.add_argument("-e", "--epochs", required=False, help="Epochs Used")

    args = parser.parse_args()

    # python main.py -w2v -n testing -lr 0.1 -bs 64 -e 2 -t 0 -w 5 -msl 128
    if args.word2vec or args.fasttext:
        name = args.name
        lr = float(args.learningrate)
        batch_size = int(args.batchsize)
        maxlen = int(args.maxseqlen)
        epochs = int(args.epochs)
        types = int(args.type)
        window = int(args.window)
        print("Config:")
        print(f"Types : {types}")
        print(f"Window : {window}")
        print(f"Epochs : {epochs}")
        print(f"Batch_size : {batch_size}")
        print(f"Learning rate : {lr}")
        print(f"Max Length : {maxlen}")

        if args.fasttext:
            main(
                config={
                    "window": window,
                    # "sg": types,  
                    "min_count":1,
                    "workers": -1,
                },
                types="ft",
                learning_rate=lr,
                batch_size=batch_size,
                max_length=maxlen,
                detail=name,
            )
        else:  # word2vec (default)
            main(
                config={
                    "window": window,
                    # "sg": types,  
                    "min_count":1,
                    "workers": -1,
                },
                types="w2v",
                learning_rate=lr,
                batch_size=batch_size,
                max_length=maxlen,
                detail=name,
            )

    elif args.bert:
        pass
        # from src.word_embedding_with_context.bert import main_bert

        # epochs = int(args.epochs)
        # lr = float(args.learningrate)
        # batch_size = int(args.batchsize)
        # maxlen = int(args.maxseqlen)
        # name = args.name
        # main_bert(
        #     name=name,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     learning_rate=lr,
        #     max_seq_len=maxlen,
        # )

    elif args.indobert:
        pass
        # from src.word_embedding_with_context.indobert import main_indobert

        # epochs = int(args.epochs)
        # lr = float(args.learningrate)
        # batch_size = int(args.batchsize)
        # maxlen = int(args.maxseqlen)

        # main_indobert(
        #     epochs=epochs, batch_size=batch_size, learning_rate=lr, max_seq_len=maxlen
        # )
