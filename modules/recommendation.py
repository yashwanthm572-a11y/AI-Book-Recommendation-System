import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BookRecommender:

    def __init__(self, books):

        self.books = books.copy()

        self.books["features"] = (
            self.books["genre"].fillna("") + " " +
            self.books["author"].fillna("") + " " +
            self.books["description"].fillna("")
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.vectors = self.vectorizer.fit_transform(
            self.books["features"]
        )

        self.similarity = cosine_similarity(
            self.vectors
        )

    def recommend(self, book_title, count=5):

        matches = self.books[
            self.books["title"] == book_title
        ]

        if matches.empty:
            return pd.DataFrame()

        index = matches.index[0]

        scores = list(
            enumerate(self.similarity[index])
        )

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        scores = [
            item for item in scores
            if item[0] != index
        ][:count]

        results = []

        for book_index, score in scores:

            book = self.books.loc[book_index]

            results.append({
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "rating": book["rating"],
                "description": book["description"],
                "similarity": round(float(score) * 100, 2)
            })

        return pd.DataFrame(results)