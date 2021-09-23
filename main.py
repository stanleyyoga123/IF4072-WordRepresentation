import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    if args.bert:
        from src.word_embedding_with_context.bert import main_bert

        epochs = int(args.epochs)
        lr = float(args.learningrate)
        batch_size = int(args.batchsize)
        maxlen = int(args.maxseqlen)
        name = args.name
        main_bert(
            name=name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_len=maxlen,
        )

    elif args.indobert:
        from src.word_embedding_with_context.indobert import main_indobert

        epochs = int(args.epochs)
        lr = float(args.learningrate)
        batch_size = int(args.batchsize)
        maxlen = int(args.maxseqlen)

        main_indobert(
            epochs=epochs, batch_size=batch_size, learning_rate=lr, max_seq_len=maxlen
        )
