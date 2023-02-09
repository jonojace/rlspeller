import torch
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import os
from fastpitch.common.text.text_processing import TextProcessor

class SpellerDataset(torch.utils.data.Dataset):
    """
    Dataset for training a speller model.  
    The dataset contains a list of words and their word-aligned mel spectrograms.

        1) loads word + word-aligned mel spec for all words in a wordlist
        2) converts text to sequences of one-hot vectors (corresponding to grapheme indices in fastpitch)
        3) returns a dict containing: 
            - word
            - encoded word
            - mel filepath
            - mel
            That is used to train the word-level speller model
    """

    def __init__(
            self,
            wordaligned_speechreps_dir,  # path to directory that contains folders of word aligned speech reps
            wordlist,  # txt file for the words to include speech reps from
            max_examples_per_wordtype=None,
            text_cleaners=["lowercase_no_punc"],
            symbol_set="english_pad_lowercase_nopunc",
            add_spaces=False,
            eos_symbol=" ",
            **kwargs,
    ):
        # load wordlist as a python list
        if type(wordlist) == str:
            if wordlist.endswith('.json'):
                with open(wordlist) as f:
                    wordlist = json.load(f)
            else:
                with open(wordlist) as f:
                    wordlist = f.read().splitlines()
        elif type(wordlist) == list:
            pass  # dont need to do anything, already in expected form
        elif type(wordlist) == set:
            wordlist = list(wordlist)

        wordlist = sorted(wordlist)

        # create list of all word tokens and their word aligned speech reps
        self.word_freq = Counter()
        self.token_and_melfilepaths = []
        print("Initialising respeller dataset")
        for word in tqdm(wordlist):
            # find all word aligned mels for the word
            word_dir = os.path.join(wordaligned_speechreps_dir, word)
            mel_files = os.listdir(word_dir)
            if max_examples_per_wordtype:
                mel_files = mel_files[:max_examples_per_wordtype]
            for mel_file in mel_files:
                mel_file_path = os.path.join(word_dir, mel_file)
                self.token_and_melfilepaths.append((word, mel_file_path))
                self.word_freq[word] += 1

        self.tp = TextProcessor(symbol_set, text_cleaners, add_spaces=add_spaces, eos_symbol=eos_symbol)

    def get_mel(self, filename):
        return torch.load(filename)

    def encode_text(self, text):
        """encode raw text into indices defined by grapheme embedding table of the TTS model"""
        return torch.IntTensor(self.tp.encode_text(text))

    def decode_text(self, encoded):
        if encoded.dim() == 1:
            decodings = [self.tp.id_to_symbol[id] for id in encoded.tolist()]
        else:
            decodings = []
            for batch_idx in range(encoded.size(0)):
                decodings.append(''.join(self.tp.id_to_symbol[idx] for idx in encoded[batch_idx].tolist()))
        return decodings

    def __getitem__(self, index):
        word, mel_filepath = self.token_and_melfilepaths[index]
        encoded_word = self.encode_text(word)
        mel = self.get_mel(mel_filepath)

        return {
            'word': word,
            'encoded_word': encoded_word,
            'mel_filepath': mel_filepath,
            'mel': mel,
        }
