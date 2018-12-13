#!/usr/bin/env python
import argparse
import os
from six.moves.urllib import request
import zipfile
import subprocess


"""Download the MSCOCO dataset (images and captions)."""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data',
                        help='Target MSCOCO dataset root directory')
    parser.add_argument('--only-anno', action="store_true",
                        help='Whether to download annotation data only')
    args = parser.parse_args()

    if args.only_anno:
        urls = [
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            'https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.2.tar.gz'
        ]
    else:
        urls = [
            'http://images.cocodataset.org/zips/train2014.zip',
            'http://images.cocodataset.org/zips/val2014.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            'https://github.com/STAIR-Lab-CIT/STAIR-captions/raw/master/stair_captions_v1.2.tar.gz'
        ]

    try:
        os.makedirs(args.out)
    except OSError:
        raise OSError(
            "'{}' already exists, delete it and try again".format(args.out))

    for url in urls:
        print('Downloading {}...'.format(url))

        # Download the file
        file_name = os.path.basename(url)
        dst_file_path = os.path.join(args.out, file_name)
        request.urlretrieve(url, dst_file_path)

        if os.path.splitext(dst_file_path)[1] == ".zip":
            # Unzip the file
            zf = zipfile.ZipFile(dst_file_path)
            for name in zf.namelist():
                dirname, filename = os.path.split(name)
                if not filename == '':
                    zf.extract(name, args.out)
        else:
            # unfreeze tar.gz file 
            cmd = "tar -zxvf {} -C {}".format(dst_file_path, os.path.join(args.out, "annotations"))
            subprocess.call(cmd, shell=True)

        # Remove the zip file since it has been extracted
        os.remove(dst_file_path)
