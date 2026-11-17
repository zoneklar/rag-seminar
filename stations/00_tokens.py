"""
Station 00: Was sind Tokens?

Lernziel:
Verstehen, wie ein Sprachmodell Text in Tokens zerlegt.

Start:
uv run python -m stations.00_tokens
"""

import tiktoken

# 1. Text in Tokens zerlegen

text = "Hello, I am learning about RAG!"

encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)

print("Text:", text)
print("Token-IDs:", tokens)


# 2. Einzelne Tokens sichtbar machen
print("Einzelne Tokens:")
for token in tokens:
    print(token, ":", f"'{encoding.decode([token])}'")


# 3. Wörter und Tokens vergleichen

text = "Retrieval-Augmented Generation ist spannend!"
tokens = encoding.encode(text)
print("\nText:", text)
print("Anzahl Wörter:", len(text.split()))
print("Anzahl Tokens:", len(tokens))

for token in tokens:
    print(token, ":", f"'{encoding.decode([token])}'")
#
# Tipps:
# text.split()
# len()
