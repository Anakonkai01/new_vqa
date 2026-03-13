"""Vocabulary class for the VQA project.

Handles tokenization, numericalization, and serialization of the token
vocabulary used by both the question encoder and the answer/caption decoder.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import List

import nltk


class Vocabulary:
    """Maps words to integer indices and back.

    Special token layout (always fixed, regardless of training data):

        Index 0 : <pad>      – padding; ignored by CrossEntropyLoss(ignore_index=0)
        Index 1 : <start>    – begin-of-sequence token
        Index 2 : <end>      – end-of-sequence token
        Index 3 : <unk>      – out-of-vocabulary token
        Index 4 : <task_vqa> – multi-task discriminator: VQA mode
        Index 5 : <task_cap> – multi-task discriminator: captioning mode

    Example::

        vocab = Vocabulary()
        vocab.build(["what color is the cat?"], threshold=1)
        vocab.numericalize("what color")
        # → [1, <what_idx>, <color_idx>, 2]
    """

    PAD_TOKEN:      str = "<pad>"
    START_TOKEN:    str = "<start>"
    END_TOKEN:      str = "<end>"
    UNK_TOKEN:      str = "<unk>"
    TASK_VQA_TOKEN: str = "<task_vqa>"
    TASK_CAP_TOKEN: str = "<task_cap>"

    # Insertion order is critical: index 0 must be <pad>, 1 must be <start>, etc.
    _SPECIAL_TOKENS: tuple = (
        PAD_TOKEN,
        START_TOKEN,
        END_TOKEN,
        UNK_TOKEN,
        TASK_VQA_TOKEN,
        TASK_CAP_TOKEN,
    )

    def __init__(self) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._next_idx: int = 0

        # Special tokens must occupy indices 0–5 in a fixed order so that
        # CrossEntropyLoss(ignore_index=0) and decoder <start>/<end> detection
        # always point to the correct positions.
        for token in self._SPECIAL_TOKENS:
            self._add_word(token)

    # ── Special-token index properties ──────────────────────────────────────

    @property
    def pad_idx(self) -> int:
        """Index of the <pad> token (always 0)."""
        return self.word2idx[self.PAD_TOKEN]

    @property
    def start_idx(self) -> int:
        """Index of the <start> token (always 1)."""
        return self.word2idx[self.START_TOKEN]

    @property
    def end_idx(self) -> int:
        """Index of the <end> token (always 2)."""
        return self.word2idx[self.END_TOKEN]

    @property
    def unk_idx(self) -> int:
        """Index of the <unk> token (always 3)."""
        return self.word2idx[self.UNK_TOKEN]

    # ── Internal helpers ────────────────────────────────────────────────────

    def _add_word(self, word: str) -> None:
        """Insert *word* if not already present. Internal use only."""
        if word not in self.word2idx:
            self.word2idx[word] = self._next_idx
            self.idx2word[self._next_idx] = word
            self._next_idx += 1

    # ── Public API ───────────────────────────────────────────────────────────

    def add_word(self, word: str) -> None:
        """Public alias for ``_add_word``; kept for backward compatibility.

        Args:
            word: Token string to insert.
        """
        self._add_word(word)

    def build(self, sentence_list: List[str], threshold: int = 3) -> None:
        """Build vocabulary from a corpus of sentences.

        Words are counted across all sentences. Only those with frequency
        >= *threshold* are added.

        Args:
            sentence_list: Raw sentences (questions, answers, captions).
            threshold: Minimum occurrence count to include a word.
        """
        counter: Counter = Counter()
        print(" -> Tokenizing...")
        for sentence in sentence_list:
            counter.update(self.tokenize(sentence))

        print(f" -> Filtering (threshold={threshold})...")
        for word, count in counter.most_common():
            if count >= threshold:
                self._add_word(word)
        print(f" -> Done. Vocab size: {len(self)}")

    def tokenize(self, sentence: str) -> List[str]:
        """Lowercase and tokenize a sentence using NLTK word_tokenize.

        Downloads NLTK punkt data on first call if missing.

        Args:
            sentence: Raw text string.

        Returns:
            List of lowercase string tokens.
        """
        sentence = sentence.lower().strip()
        try:
            tokens = nltk.word_tokenize(sentence)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            tokens = nltk.word_tokenize(sentence)
        return tokens

    def numericalize(self, sentence: str) -> List[int]:
        """Convert a sentence to a list of token indices.

        Output is wrapped with <start> and <end> tokens so it can be passed
        directly to the LSTM decoder as a target sequence.

        Args:
            sentence: Raw text string.

        Returns:
            ``[start_idx, tok_1, ..., tok_n, end_idx]``
        """
        tokens = self.tokenize(sentence)
        indices = [self.start_idx]
        for token in tokens:
            indices.append(self.word2idx.get(token, self.unk_idx))
        indices.append(self.end_idx)
        return indices

    def decode(self, indices: List[int], skip_special: bool = True) -> str:
        """Convert a list of token indices back to a readable string.

        Decoding stops at the first <end> token. Use this in inference and
        evaluation to convert model output tensors to text.

        Args:
            indices: Sequence of integer token IDs.
            skip_special: If True, omit <pad>, <start>, and task tokens.

        Returns:
            Space-joined string of the decoded words.
        """
        skip_ids = {
            self.pad_idx,
            self.start_idx,
            self.word2idx.get(self.TASK_VQA_TOKEN, -1),
            self.word2idx.get(self.TASK_CAP_TOKEN, -1),
        }
        words: List[str] = []
        for idx in indices:
            if idx == self.end_idx:
                break
            if skip_special and idx in skip_ids:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return " ".join(words)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the vocabulary to a JSON file.

        Args:
            path: Destination file path (e.g. ``'data/processed/vocab_answers.json'``).
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "word2idx": self.word2idx,
                    "idx2word": self.idx2word,
                    "next_idx": self._next_idx,
                },
                f,
                ensure_ascii=False,
            )

    def load(self, path: str) -> None:
        """Deserialize the vocabulary from a JSON file.

        Handles both the legacy ``'idx'`` key and the current ``'next_idx'`` key
        for backward compatibility with pre-existing vocab checkpoint files.

        Args:
            path: Source file path.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        # JSON serializes all object keys as strings; restore int keys for idx2word.
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        # Support both old key ("idx") and new key ("next_idx").
        self._next_idx = data.get("next_idx", data.get("idx", len(self.word2idx)))

    # ── Dunder helpers ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        # Use the actual dict length, not a counter, to avoid drift.
        return len(self.word2idx)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"
