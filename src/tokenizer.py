import re

class SimpleTokenizer:
    def __init__(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        self.vocab = sorted(set(tokens))
        self.vocab.extend(["<|unk|>", "<|endoftext|>"])
        
        self.str_to_int = {s:i for i,s in enumerate(self.vocab)}
        self.int_to_str = {i:s for s,i in self.str_to_int.items()}

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [self.str_to_int.get(t, self.str_to_int["<|unk|>"]) for t in tokens]

    def decode(self, ids):
        return " ".join([self.int_to_str[i] for i in ids])