"""Retrieval déterministe : recherche hybride (BM25 + dense) + reranking.

Interface VectorStore abstraite ; implémentation par défaut Qdrant (souverain).
Embeddings BioLORD calculés en local. Sortie : top-k candidats à haut signal.
"""
