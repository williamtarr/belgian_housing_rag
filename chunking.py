from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Read the extracted text
with open("extracted_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create text splitter with better settings for accounting documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Larger chunks to preserve more context
    chunk_overlap=200,  # Significant overlap to maintain context between chunks
    separators=[
        "\n\nArt.",  # Split on article boundaries
        "\n\n",      # Then on double newlines
        "\n",        # Then on single newlines
        ". ",        # Then on sentences
        " "          # Finally on words as last resort
    ],
    length_function=len,
    is_separator_regex=False
)

# Split text into chunks
chunks = splitter.split_text(text)

# Convert chunks to structured format with better titles
structured_chunks = []
for chunk in chunks:
    # Try to find a meaningful title from the first line
    lines = chunk.strip().split('\n')
    first_line = lines[0].strip()
    
    # Use first line if it's short enough, otherwise look for common section markers
    if len(first_line) < 50:
        title = first_line
    elif "Art." in chunk[:50]:
        title = chunk[chunk.find("Art."):].split('\n')[0]
    elif any(marker in chunk[:100] for marker in ["CHAPITRE", "Section", "Annexe"]):
        title = next(line for line in lines if any(marker in line for marker in ["CHAPITRE", "Section", "Annexe"]))
    else:
        # Use first meaningful text as title
        title = first_line[:50] + "..."

    structured_chunks.append({
        "title": title,
        "content": chunk.strip()
    })

# Save chunks to JSON file
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(structured_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(structured_chunks)} sections to chunks.json")

# Now chunks contains your text split into manageable segments

