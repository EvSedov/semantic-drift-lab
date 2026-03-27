"""
SVD Embedder — шаг 1 пайплайна Semantic Drift Lab.

Преобразует коллекцию текстов в компактные векторы через TF-IDF + TruncatedSVD
(Latent Semantic Analysis), чтобы анализировать геометрию корпуса в
интерпретируемом низкоразмерном пространстве.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class SVDEmbedder:
    """TF-IDF + TruncatedSVD для получения плотных векторных представлений текстов."""

    def __init__(self, n_components: int = 10, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.explained_variance_ratio_: float = 0.0

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """
        Обучает и трансформирует список текстов в матрицу (n_texts, n_components).
        Автоматически ограничивает n_components если документов мало.
        """
        n = len(texts)
        # SVD требует n_components < min(n_samples, n_features)
        safe_k = min(self.n_components, n - 1, 50)
        if safe_k != self.n_components:
            self._svd = TruncatedSVD(n_components=safe_k, random_state=self.random_state)

        tfidf_matrix = self._vectorizer.fit_transform(texts)
        embedded = self._svd.fit_transform(tfidf_matrix)
        self.explained_variance_ratio_ = float(self._svd.explained_variance_ratio_.sum())

        # L2-нормализация — стабилизирует kNN
        return normalize(embedded, norm="l2")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Трансформирует новые тексты в уже обученное пространство."""
        tfidf_matrix = self._vectorizer.transform(texts)
        embedded = self._svd.transform(tfidf_matrix)
        return normalize(embedded, norm="l2")
