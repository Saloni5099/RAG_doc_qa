# step1_load.py
# Goal: just read a text file into memory and confirm it works

# "r" — means read mode (we're only reading, not writing/changing the file)

# encoding="utf-8" — tells Python how to interpret the text characters (utf-8 is the standard, handles almost all languages/symbols safely)
with open("documents/sample.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded {len(text)} characters")

 # f"..." — this is called an f-string. It lets you insert a variable's value directly inside a text message using {}
print(text[:200])  # print first 200 characters to sanity check


# Why this step matters for RAG
# Before a computer can search or answer questions about your document, it first needs the document as data inside the program — not just sitting as a file on your hard drive. This script is that very first bridge: file on disk → text inside Python, ready to be processed in the next steps (chunking, embedding, etc.).
# Did running python step1_load.py show you the character count and preview text? 