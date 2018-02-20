from __future__ import absolute_import
from __future__ import division

import argparse
import sys
from data_util import load_and_preprocess_data, load_embeddings, read_conll, ModelHelper
from q1_window import WindowModel
from q1_window import Config
import tensorflow as tf

def do_test1(_):
    print("Hello, I am inside do_test1")

def do_test2(args):
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    Config.embed_size = embeddings.shape[1]
    # print(embed_size)
    # print(train)
    #
    with tf.Graph().as_default():
        model = WindowModel(helper, Config, embeddings)
        init = tf.global_variables_initializer()
        saver = None
        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

if __name__ == "__main__":
    print("Inside Main method")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("echo1")
    # args = parser.parse_args()
    # print(args.echo1)

    parser = argparse.ArgumentParser(description="NER Models")

    subparsers = parser.add_subparsers()
    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_test1)


    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_test2)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)