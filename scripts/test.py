import sys
import os

# Add the project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.tokenizer import BPETokenizer

# Sample corpus
corpus = "hello world hello machine learning"

# Initialize and build tokenizer
tokenizer = BPETokenizer(vocab_size=50)
tokenizer.build_vocab(corpus)

# Encode and decode
encoded = tokenizer.encode("hello machine")
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
