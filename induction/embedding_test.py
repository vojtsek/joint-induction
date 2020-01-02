import sys

from .embeddings import Embeddings

embeddings = Embeddings(sys.argv[1])
print('loaded')
while True:
    w1 = input('w1 ').strip()
    w2 = input('w2 ').strip()
    e1 = embeddings.embed_tokens([w1])
    e2 = embeddings.embed_tokens([w2])
    print(embeddings.embedding_similarity(e1, e2))
