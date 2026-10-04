import re
from SimpleTokenizerV1 import SimpleTokenizerV1
from SimpleTokenizerV2 import SimpleTokenizerV2

with open("../the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

tokenizerV1 = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know, "
        Mrs. Gisburn said with pardonable pride."""

print("Raw text:", text)
idsV1 = tokenizerV1.encode(text)
print("TokenizerV1 IDs:", idsV1)
print("TokenizerV1 decoded IDs:", tokenizerV1.decode(idsV1))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text3 = " <|endoftext|> ".join((text1, text2))

tokenizerV2 = SimpleTokenizerV2(vocab)
print("\nRaw text:", text3)
idsV2 = tokenizerV2.encode(text3)
print("TokenizerV2 IDs:", idsV2)
print("TokenizerV1 decoded IDs:", tokenizerV2.decode(idsV2))
