## Overall Used Libraries / Technologies

![Technologies Used](https://github.com/user-attachments/assets/bdaa3248-423f-4cf3-8dba-ff81ed7197fe)

---

## Detailed Design Decisions

### Text Extraction

The primary library used is **PyMuPDF**, with a fallback to **PyPDF2**.

- **PyMuPDF** is chosen because of its excellent support for **Unicode and multilingual text**, which is critical for processing Bengali-language documents.
- If PyMuPDF fails, the system falls back to **PyPDF2**, which offers a basic but reliable method for PDF parsing.

---

### Chunking Strategy

This system uses a **sentence-based, fixed word-count chunking** approach with overlap.

**Why this approach is effective for semantic retrieval:**

1. **Sentence-level chunking** ensures each chunk contains a complete thought or unit of meaning.
2. It avoids breaking sentences in the middle, which can lead to loss of context.
3. **Overlap between chunks** helps maintain continuity and improves retrieval accuracy.

**Parameters used:**

- Chunk size: 300 words  
- Overlap: 150 words

This strategy is more effective than arbitrary character or word splits, especially when dealing with natural language in Bengali and English.
