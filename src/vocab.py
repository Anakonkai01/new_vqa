# File: src/vocab.py
import json
import re
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0 # tracking index 
        
        # 4 special tokens required for generative decoding
        self.pad_token = "<pad>"    # used to pad batches to equal length
        self.start_token = "<start>"# signals start of sequence
        self.end_token = "<end>"    # signals end of sequence
        self.unk_token = "<unk>"    # out-of-vocabulary token
        
        self.add_word(self.pad_token)   # idx 0
        self.add_word(self.start_token) # idx 1
        self.add_word(self.end_token)   # idx 2
        self.add_word(self.unk_token)   # idx 3

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build(self, sentence_list, threshold=3):
        """
        Build the vocabulary from a list of sentences.
        threshold: minimum occurrence count to keep a word.
        """
        counter = Counter()
        print(" -> Tokenizing...")
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        print(f" -> Filtering (threshold={threshold})...")
        for word, count in counter.items():
            if count >= threshold:
                self.add_word(word)
        print(f" -> Done. Vocab size: {len(self.word2idx)}")

    def tokenize(self, sentence):
        """Lowercase and tokenize a sentence using simple word splitting."""
        sentence = sentence.lower().strip()
        # Keep alphanumeric characters, strip punctuation
        tokens = re.findall(r"\w+", sentence)
        return tokens

    def numericalize(self, sentence):
        """Convert text to a list of token indices."""
        tokens = self.tokenize(sentence)
        result = [self.word2idx[self.start_token]]
        for token in tokens:
            # Use known word ID if in vocab, otherwise use <unk>
            word_id = self.word2idx.get(token, self.word2idx[self.unk_token])
            result.append(word_id)
        result.append(self.word2idx[self.end_token])
        return result

    def __len__(self):
        return self.idx

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'idx': self.idx
            }, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.word2idx = data['word2idx']
            # JSON keys are always strings; convert back to int
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            self.idx = data['idx']