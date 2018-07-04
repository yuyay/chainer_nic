#!/usr/bin/env python
#! encoding=UTF-8
import argparse
import glob
import os

import numpy as np
from PIL import Image
import json

import chainer
from chainer import serializers

import datasets
from model import ImageCaptionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str,
                        help='Image path')
    parser.add_argument('--img-dir', type=str,
                        help='Image directory path, instead of a single image')
    parser.add_argument('--model', type=str, default='result/model_1000',
                        help='Trained model path')
    parser.add_argument('--dataset-name', type=str, default='mscoco', choices=["mscoco", "stair_captions"],
                        help='MSCOCO dataset root directory')
    parser.add_argument('--mscoco-root', type=str, default='data',
                        help='MSOCO dataset root directory')
    parser.add_argument('--rnn', type=str, default='nsteplstm',
                        choices=['nsteplstm', 'lstm'],
                        help='Language model layer type')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max-caption-length', type=int, default=30,
                        help='Maximum caption length generated')
    parser.add_argument('--out', type=str,
                        help='Json file to save predicted captions',
                        default='prediction.json')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Minibatch size')
    args = parser.parse_args()

    # Set path to annotation files
    if args.dataset_name == "mscoco":
        train_anno = "annotations/captions_train2014.json"
        val_anno = "annotations/captions_val2014.json"
    elif args.dataset_name == "stair_captions":
        train_anno = "annotations/stair_captions_v1.2_train_tokenized.json"
        val_anno = "annotations/stair_captions_v1.2_val_tokenized.json"

    # Load the dataset to obtain the vocabulary, which is needed to convert
    # predicted tokens into actual words
    train, _ = datasets.get_mscoco(
        args.mscoco_root, train_anno=train_anno, val_anno=val_anno, dataset_name=args.dataset_name)
    vocab = train.vocab
    ivocab = {v: k for k, v in vocab.items()}

    model = ImageCaptionModel(len(train.vocab), rnn=args.rnn)
    serializers.load_npz(args.model, model)

    if args.img_dir:  # Read all images in directory
        img_paths = [
            i for i in glob.glob(os.path.join(args.img_dir, '*')) if
            i.endswith(('png', 'jpg'))]
        img_paths = sorted(img_paths)
    else:  # Load a single image
        img_paths = [args.img]

    if not img_paths:
        raise IOError('No images found for the given path')

    img_paths = np.random.permutation(img_paths)
    results = dict()
    for i in range(0, len(img_paths), args.batch_size):
        img_paths_sub = img_paths[i:i + args.batch_size]
        imgs = []
        for img_path in img_paths_sub:
            img = Image.open(img_path)
            img = model.prepare(img)
            imgs.append(img)
        imgs = np.asarray(imgs)

        if args.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()
            imgs = chainer.backends.cuda.to_gpu(imgs)

        bos = vocab['<bos>']
        eos = vocab['<eos>']
        with chainer.using_config('train', False), \
                chainer.no_backprop_mode():
            captions = model.predict(
                imgs, bos=bos, eos=eos, max_caption_length=args.max_caption_length)
        captions = chainer.backends.cuda.to_cpu(captions)

        # Print the predicted captions
        file_names = [os.path.basename(path) for path in img_paths_sub]
        max_length = max(len(name) for name in file_names)
        for file_name, caption in zip(file_names, captions):
            caption = ' '.join(ivocab[token] for token in caption)
            caption = caption.replace('<bos>', '')
            end = caption.find('<eos>')
            caption = caption[:end].strip()
            # caption = caption.replace('<bos>', '').replace('<eos>', '').strip()
            results[file_name] = caption
            print(('{0:' + str(max_length) + '} {1}').format(file_name, caption))

    # Save captions
    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)
    
        

if __name__ == '__main__':
    main()
