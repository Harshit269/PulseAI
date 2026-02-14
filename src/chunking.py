import os

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


def main():
    file_path = os.path.join("data", "sample.txt")

    text = load_text(file_path)
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk)
        print("-" * 50)


if __name__ == "__main__":
    main()
