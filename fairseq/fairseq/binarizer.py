# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import os

from fairseq.tokenizer import tokenize_line


def safe_readline(f):
    """
    Safely writeline until the file - like read.

    Args:
        f: (todo): write your description
    """
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        """
        Binarize a text file.

        Args:
            filename: (str): write your description
            dict: (todo): write your description
            consumer: (todo): write your description
            tokenize: (int): write your description
            tokenize_line: (str): write your description
            append_eos: (int): write your description
            reverse_order: (bool): write your description
            offset: (float): write your description
            end: (int): write your description
        """
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            """
            Replaced the consumer.

            Args:
                word: (str): write your description
                idx: (todo): write your description
            """
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        """
        Alignsar file.

        Args:
            filename: (str): write your description
            alignment_parser: (todo): write your description
            consumer: (todo): write your description
            offset: (todo): write your description
            end: (todo): write your description
        """
        nseq = 0

        with open(filename, 'r') as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        """
        Find the number of filename.

        Args:
            filename: (str): write your description
            num_chunks: (int): write your description
        """
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
