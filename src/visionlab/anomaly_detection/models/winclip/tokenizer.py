# src/visionlab/anomaly_detection/models/winclip/tokenizer.py: Lazy independent CLIP BPE tokenizer
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt
# derived from open_clip (MIT license)
# The bundled BPE vocabulary was created by OpenAI CLIP and distributed with open_clip.

import gzip
import html
import os
from functools import lru_cache

import torch


@lru_cache()
def bytes_to_unicode():
    """Map UTF-8 bytes reversibly to non-whitespace Unicode symbols."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes = values[:]
    count = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            codes.append(256 + count)
            count += 1
    return dict(zip(values, map(chr, codes)))


def get_pairs(word):
    """Return adjacent symbol pairs for BPE merging."""
    return set(zip(word, word[1:]))


def basic_clean(text):
    """Unescape HTML twice and strip surrounding whitespace."""
    # ftfy.fix_text is omitted because WinCLIP prompts are fixed ASCII strings.
    return html.unescape(html.unescape(text)).strip()


class SimpleTokenizer:
    """Tokenize WinCLIP prompts with the original lower-cleaning BPE path."""

    def __init__(self, bpe_path=None, context_length=77):
        import regex

        if bpe_path is None:
            bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in self.byte_encoder.items()}
        with gzip.open(bpe_path) as stream:
            merges = stream.read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(self.byte_encoder.values())
        vocab += [value + "</w>" for value in vocab]
        vocab += ["".join(merge) for merge in merges]
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {value: key for key, value in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {token: token for token in special_tokens}
        self.pat = regex.compile(
            "|".join(special_tokens) + r"|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )
        self.whitespace = regex.compile(r"\s+")
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[token] for token in special_tokens]
        self.sot_token_id, self.eot_token_id = self.all_special_ids
        self.context_length = context_length

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    new_word.extend(word[index:])
                    break
                new_word.extend(word[index:next_index])
                index = next_index
                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        text = self.whitespace.sub(" ", basic_clean(text)).strip().lower()
        tokens = []
        for token in self.pat.findall(text):
            token = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            tokens.extend(self.encoder[value] for value in self.bpe(token).split(" "))
        return tokens

    def decode(self, tokens):
        text = "".join(self.decoder[token] for token in tokens)
        return bytearray(self.byte_decoder[value] for value in text).decode(
            "utf-8", errors="replace"
        ).replace("</w>", " ")

    def __call__(self, texts, context_length=None):
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length
        assert context_length, "Please set a valid context length"
        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        for index, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[index, :len(tokens)] = torch.tensor(tokens)
        return result


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Open the vocabulary only when tokenization is first requested."""
    return SimpleTokenizer()


def tokenize(texts, context_length=77):
    """Return zero-padded CLIP tokens with EOT-preserving truncation."""
    return _get_tokenizer()(texts, context_length=context_length)
