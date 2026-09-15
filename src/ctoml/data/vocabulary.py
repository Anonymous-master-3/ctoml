from collections.abc import Iterable


class Vocabulary:

    PAD, BOS, EOS, UNK = 0, 1, 2, 3
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def __init__(self, tokens: list[str], mode: str = 'word'):
        if mode not in ('word', 'character'):
            raise ValueError('mode must be word or character')
        if len(set(tokens)) != len(tokens):
            raise ValueError('vocabulary contains duplicate lexical tokens')
        self.mode = mode
        self.tokens = list(tokens)
        self.token_to_id = {token: i + 4 for i, token in enumerate(tokens)}

    @classmethod
    def from_texts(cls, texts: Iterable[str], mode: str = 'word'):
        vocab = cls([], mode)
        return cls(sorted({token for text in texts for token in vocab.tokenize(text)}), mode)

    def tokenize(self, text: str) -> list[str]:
        return text.split() if self.mode == 'word' else list(text)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.UNK) for token in self.tokenize(text)]

    def decode(self, ids: Iterable[int]) -> str:
        words = []
        for value in ids:
            value = int(value)
            if value == self.EOS:
                break
            if value in (self.PAD, self.BOS):
                continue
            words.append(self.tokens[value - 4] if 4 <= value < len(self) else '<unk>')
        return (' ' if self.mode == 'word' else '').join(words)

    def __len__(self):
        return len(self.tokens) + 4

    def to_dict(self):
        return {'mode': self.mode, 'tokens': self.tokens, 'special_ids': {'pad': 0, 'bos': 1, 'eos': 2, 'unk': 3}}

    @classmethod
    def from_dict(cls, data):
        expected = {'pad': 0, 'bos': 1, 'eos': 2, 'unk': 3}
        if data.get('special_ids', expected) != expected:
            raise ValueError('incompatible special token IDs')
        return cls(data['tokens'], data['mode'])
