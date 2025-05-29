from sentence_transformers import SentenceTransformer
from similarity import cosine_similarity
translated = ["This is an example sentence", "Esto es una frase ejemplar"]
paraphrased = ["This is an example sentence", "Take an example like the following"]
unrelated = ["This is an example sentence", "Tiene sabor strawberry cheese cake"]

model = SentenceTransformer('sentence-transformers/LaBSE')
for key, to_embed in {'translated': translated, 'paraphrased': paraphrased, 'unrelated': unrelated}.items():
    embeddings = model.encode(to_embed)
    similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
    print(f"Cosine similarity of {key} phrases: {similarity}")
