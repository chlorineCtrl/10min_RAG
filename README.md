# Overall used Libraries/Technologies :

![Untitled-2025-01-21-1548](https://github.com/user-attachments/assets/bdaa3248-423f-4cf3-8dba-ff81ed7197fe)


# Detail :

**Text Extraction: ** The primary library used is `PyMuPDF` with a fallback to `PyPDF2`. `PyMuPDF` is known for its excellent Unicode and multilingual text extraction capabilities, which is crucial for Bengali. `PyPDF2` is used as a backup in case `PyMuPDF` fails.
**chunking strategy: ** My model uses sentence-based, fixed word-count chunking with overlap strategy. This works best for semantic retrival because : 
                                                                                                                               i) By chunking at the sentence level, each chunk is likely to contain a complete thought or idea.
                                                                                                                               ii) This is better than arbitrary character or word splits, which can break sentences and lose context.
                                                                                                                               
