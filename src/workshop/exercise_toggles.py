# Created by Nir Ben-Zvi
# Workshop Exercise Toggles
# me@nirbnzvi.com

"""
Per-Exercise Solution Toggles.

Set each to False for participants to implement, True to use working solutions.
This allows granular control over which exercises are active.

Exercise descriptions:
- SIMILARITY: cosine_similarity() and get_top_k() for Phase 2 engine
- RERANKING: rerank() for Phase 3 two-stage retrieval
- PROMPTING: build_full_prompt() for RAG system prompt design
- SEGMENTING: segment_by_time_gaps() and chunk_segments() for Phase 4 chunker

This file is intentionally minimal (no imports) to avoid circular dependencies.
"""

USE_SIMILARITY_SOLUTION = True
USE_RERANKING_SOLUTION = True
USE_PROMPTING_SOLUTION = True
USE_SEGMENTING_SOLUTION = True
