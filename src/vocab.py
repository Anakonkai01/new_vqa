# File: src/vocab.py
import json
import re
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0 # tracking index 
        
        # 4 Token đặc biệt bắt buộc cho bài toán sinh (Generative)
        self.pad_token = "<pad>"    # Để lấp đầy cho batch bằng nhau
        self.start_token = "<start>"# Báo hiệu bắt đầu câu
        self.end_token = "<end>"    # Báo hiệu kết thúc câu
        self.unk_token = "<unk>"    # Từ lạ chưa học bao giờ
        
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
        Xây dựng từ điển.
        threshold: Số lần xuất hiện tối thiểu để được giữ lại.
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
        """Chuyển câu thành chữ thường và tách từ đơn giản"""
        sentence = sentence.lower().strip()
        # Regex này giữ lại chữ cái và số, bỏ dấu câu
        tokens = re.findall(r"\w+", sentence) 
        return tokens

    def numericalize(self, sentence):
        """Biến đổi Text -> List số (Indices)"""
        tokens = self.tokenize(sentence)
        result = [self.word2idx[self.start_token]]
        for token in tokens:
            # Nếu từ có trong từ điển thì lấy ID, không thì lấy ID của <unk>
            word_id = self.word2idx.get(token, self.word2idx[self.unk_token])
            result.append(word_id)
        result.append(self.word2idx[self.end_token])
        return result

    def __len__(self): # =)) ao ma 
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
            # JSON key luôn là string, cần convert lại key thành int
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            self.idx = data['idx']