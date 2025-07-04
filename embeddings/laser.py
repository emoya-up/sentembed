from laser_encoders import LaserEncoderPipeline

encoder = LaserEncoderPipeline(lang="eng-Latn")
embeddings = encoder.encode_sentences(["H!", "This is a sentence encoder."])
print(embeddings.shape)  # (2, 1024)