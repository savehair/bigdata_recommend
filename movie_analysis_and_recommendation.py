import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_multivalue_column(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )


def preprocess_imdb(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    numeric_cols = [
        "Rank",
        "Year",
        "Runtime (Minutes)",
        "Rating",
        "Votes",
        "Revenue (Millions)",
        "Metascore",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    text_fill_cols = ["Genre", "Description", "Director", "Actors", "Title"]
    for col in text_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # 去重：优先按标题+年份去重，其次全行去重
    if {"Title", "Year"}.issubset(df.columns):
        df = df.sort_values("Rank", na_position="last").drop_duplicates(
            subset=["Title", "Year"], keep="first"
        )
    df = df.drop_duplicates().reset_index(drop=True)

    # 为可用性保留缺失收入，但其它关键数值若缺失则用于统计时 dropna
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    directors = parse_multivalue_column(df["Director"]) if "Director" in df else pd.Series(dtype=str)
    actors = parse_multivalue_column(df["Actors"]) if "Actors" in df else pd.Series(dtype=str)
    genres = parse_multivalue_column(df["Genre"]) if "Genre" in df else pd.Series(dtype=str)
    return {
        "movie_count": int(len(df)),
        "director_count": int(directors.nunique()),
        "actor_count": int(actors.nunique()),
        "genre_count": int(genres.nunique()),
    }


def plot_votes_distribution(df: pd.DataFrame, outdir: Path) -> dict:
    votes = df["Votes"].dropna()
    stats = {
        "votes_mean": float(votes.mean()),
        "votes_max": float(votes.max()),
        "votes_min": float(votes.min()),
    }
    plt.figure(figsize=(8, 5))
    sns.histplot(votes, bins=30, kde=True)
    plt.title("Votes Distribution")
    plt.xlabel("Votes")
    plt.tight_layout()
    plt.savefig(outdir / "votes_distribution.png", dpi=150)
    plt.close()
    return stats


def revenue_rating_relation(df: pd.DataFrame, outdir: Path) -> dict:
    sub = df[["Revenue (Millions)", "Rating"]].dropna()
    corr = sub["Revenue (Millions)"].corr(sub["Rating"])
    plt.figure(figsize=(7, 5))
    sns.regplot(data=sub, x="Revenue (Millions)", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Revenue vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "revenue_rating_relation.png", dpi=150)
    plt.close()
    return {"revenue_rating_corr": float(corr)}


def genre_analysis(df: pd.DataFrame, outdir: Path) -> dict:
    genre_series = parse_multivalue_column(df["Genre"])
    genre_counts = genre_series.value_counts()

    top_two = genre_counts.head(2).to_dict()

    runtime_bin = pd.cut(
        df["Runtime (Minutes)"],
        bins=[-np.inf, 90, 120, np.inf],
        labels=["short", "medium", "long"],
        right=False,
    )
    df2 = df.copy()
    df2["runtime_bin"] = runtime_bin

    exploded = df2.assign(Genre=df2["Genre"].str.split(",")).explode("Genre")
    exploded["Genre"] = exploded["Genre"].str.strip()

    grp = exploded.dropna(subset=["runtime_bin", "Rating"]).groupby(["runtime_bin", "Genre"])["Rating"].mean()
    lowest_per_bin = grp.groupby(level=0).idxmin().to_dict()
    lowest_per_bin = {
        str(k): {"genre": v[1], "avg_rating": float(grp[v])}
        for k, v in lowest_per_bin.items()
    }

    plt.figure(figsize=(10, 5))
    genre_counts.head(15).plot(kind="bar")
    plt.title("Top 15 Genres by Count")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "genre_count_top15.png", dpi=150)
    plt.close()

    rev_genre = exploded.dropna(subset=["Revenue (Millions)"]).groupby("Genre")["Revenue (Millions)"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    rev_genre.head(15).plot(kind="bar")
    plt.title("Top 15 Genres by Average Revenue")
    plt.ylabel("Avg Revenue (Millions)")
    plt.tight_layout()
    plt.savefig(outdir / "genre_avg_revenue_top15.png", dpi=150)
    plt.close()

    return {
        "top_two_genres": top_two,
        "lowest_genre_by_runtime_bin": lowest_per_bin,
        "avg_revenue_by_genre": rev_genre.to_dict(),
    }


def director_actor_analysis(df: pd.DataFrame, outdir: Path) -> dict:
    directors = parse_multivalue_column(df["Director"])
    actors = parse_multivalue_column(df["Actors"])

    dir_top = directors.value_counts().head(1).to_dict()
    actor_top = actors.value_counts().head(1).to_dict()

    df2016 = df[df["Year"] == 2016]
    dir2016 = parse_multivalue_column(df2016["Director"]).value_counts().head(1).to_dict()
    actor2016 = parse_multivalue_column(df2016["Actors"]).value_counts().head(1).to_dict()

    # 每部电影演员数量与评分相关性
    actor_num = df["Actors"].fillna("").str.split(",").apply(lambda x: len([i for i in x if str(i).strip()]))
    sub = pd.DataFrame({"actor_count": actor_num, "Rating": df["Rating"]}).dropna()
    corr = sub["actor_count"].corr(sub["Rating"])

    plt.figure(figsize=(7, 5))
    sns.regplot(data=sub, x="actor_count", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Actor Count vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "actor_count_rating_relation.png", dpi=150)
    plt.close()

    return {
        "most_common_director": dir_top,
        "most_common_actor": actor_top,
        "most_common_director_2016": dir2016,
        "most_common_actor_2016": actor2016,
        "actor_count_rating_corr": float(corr),
    }


def year_runtime_analysis(df: pd.DataFrame, outdir: Path) -> dict:
    year_counts = df["Year"].value_counts().sort_index()
    min_year = int(year_counts[year_counts == year_counts.min()].index.min())

    plt.figure(figsize=(10, 5))
    year_counts.plot(kind="bar")
    plt.title("Movie Counts by Year")
    plt.tight_layout()
    plt.savefig(outdir / "year_count_distribution.png", dpi=150)
    plt.close()

    rt = df["Runtime (Minutes)"].dropna()
    runtime_stats = {
        "runtime_mean": float(rt.mean()),
        "runtime_max": float(rt.max()),
        "runtime_min": float(rt.min()),
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(rt, bins=30, kde=True)
    plt.title("Runtime Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "runtime_distribution.png", dpi=150)
    plt.close()

    yr_rating = df[["Year", "Rating"]].dropna()
    corr = yr_rating["Year"].corr(yr_rating["Rating"])

    plt.figure(figsize=(7, 5))
    sns.regplot(data=yr_rating, x="Year", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Year vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "year_rating_relation.png", dpi=150)
    plt.close()

    return {
        "min_movie_count_year": min_year,
        **runtime_stats,
        "year_rating_corr": float(corr),
    }


def clustering(df: pd.DataFrame, outdir: Path) -> dict:
    feats = ["Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore", "Year"]
    sub = df[feats].dropna()
    if len(sub) < 6:
        return {"cluster_error": "Not enough rows for 5 clusters"}

    scaler = StandardScaler()
    x = scaler.fit_transform(sub)

    model = KMeans(n_clusters=5, random_state=42, n_init=20)
    labels = model.fit_predict(x)
    sil = silhouette_score(x, labels)

    vis = pd.DataFrame(x[:, :2], columns=["f1", "f2"])
    vis["cluster"] = labels

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=vis, x="f1", y="f2", hue="cluster", palette="tab10")
    plt.title("KMeans Clusters (first two standardized features)")
    plt.tight_layout()
    plt.savefig(outdir / "kmeans_clusters.png", dpi=150)
    plt.close()

    return {
        "silhouette_score": float(sil),
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
    }


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="userId", columns="itemId", values="rating")


def similarity_matrix(user_item: pd.DataFrame, method: str) -> pd.DataFrame:
    mat = user_item.fillna(0)
    if method == "cosine":
        sim = cosine_similarity(mat.values)
        return pd.DataFrame(sim, index=mat.index, columns=mat.index)
    if method == "pearson":
        return user_item.T.corr(method="pearson").fillna(0)
    raise ValueError("method must be cosine or pearson")


def predict_user_item_rating(user_item: pd.DataFrame, sim_df: pd.DataFrame, user: int, item: int, k: int = 20) -> float:
    if user not in user_item.index:
        return float("nan")

    if item not in user_item.columns:
        return float("nan")

    target_sims = sim_df.loc[user].drop(index=user, errors="ignore")
    rated_users = user_item[item].dropna().index
    neigh = target_sims.loc[target_sims.index.intersection(rated_users)]

    if neigh.empty:
        return float(user_item.loc[user].mean()) if not np.isnan(user_item.loc[user].mean()) else float("nan")

    neigh = neigh.sort_values(ascending=False).head(k)
    ratings = user_item.loc[neigh.index, item]

    denom = np.abs(neigh).sum()
    if denom == 0:
        return float(ratings.mean())
    return float(np.dot(neigh.values, ratings.values) / denom)


def evaluate_user_rmse(ratings: pd.DataFrame, user: int, method: str) -> float:
    user_rated = ratings[ratings["userId"] == user]
    if len(user_rated) < 2:
        return float("nan")

    preds, truths = [], []
    for _, row in user_rated.iterrows():
        temp = ratings.drop(index=row.name)
        ui = build_user_item_matrix(temp)
        sim = similarity_matrix(ui, method)
        pred = predict_user_item_rating(ui, sim, user, row["itemId"])
        if not np.isnan(pred):
            preds.append(pred)
            truths.append(row["rating"])

    if not preds:
        return float("nan")
    return float(np.sqrt(mean_squared_error(truths, preds)))


def recommend_for_user(ratings: pd.DataFrame, user: int, method: str, topn: int = 5) -> list:
    ui = build_user_item_matrix(ratings)
    sim = similarity_matrix(ui, method)

    if user not in ui.index:
        return []

    unseen = ui.loc[user][ui.loc[user].isna()].index.tolist()
    recs = []
    for item in unseen:
        pred = predict_user_item_rating(ui, sim, user, item)
        if not np.isnan(pred):
            recs.append((int(item), float(pred)))

    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:topn]


def recommendation_analysis(ratings: pd.DataFrame) -> dict:
    ratings = ratings.copy()
    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce").astype("Int64")
    ratings["itemId"] = pd.to_numeric(ratings["itemId"], errors="coerce").astype("Int64")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["userId", "itemId", "rating"]).astype({"userId": int, "itemId": int})

    ui = build_user_item_matrix(ratings)
    sim_cos = similarity_matrix(ui, "cosine")
    sim_pea = similarity_matrix(ui, "pearson")

    pred_cos_2_7 = predict_user_item_rating(ui, sim_cos, user=2, item=7)
    pred_pea_2_7 = predict_user_item_rating(ui, sim_pea, user=2, item=7)

    rmse_cos = evaluate_user_rmse(ratings, user=2, method="cosine")
    rmse_pea = evaluate_user_rmse(ratings, user=2, method="pearson")

    best_method = "cosine" if (np.isnan(rmse_pea) or rmse_cos <= rmse_pea) else "pearson"
    recs = recommend_for_user(ratings, user=2, method=best_method, topn=5)

    return {
        "predict_user2_item7": {
            "cosine": float(pred_cos_2_7) if not np.isnan(pred_cos_2_7) else None,
            "pearson": float(pred_pea_2_7) if not np.isnan(pred_pea_2_7) else None,
        },
        "rmse_user2": {
            "cosine": float(rmse_cos) if not np.isnan(rmse_cos) else None,
            "pearson": float(rmse_pea) if not np.isnan(rmse_pea) else None,
        },
        "best_method": best_method,
        "top5_recommendations_for_user2": [
            {"itemId": item, "pred_rating": score} for item, score in recs
        ],
    }


def run(imdb_path: Path, rating_path: Path, outdir: Path) -> None:
    outdir = ensure_output_dir(outdir)

    imdb = pd.read_csv(imdb_path)
    imdb = preprocess_imdb(imdb)

    results = {
        "1_preprocess": {"rows_after_preprocess": int(len(imdb))},
        "2_exploration": {},
        "3_genre_analysis": {},
        "4_director_actor": {},
        "5_year_runtime": {},
        "6_clustering": {},
        "7_recommendation": {},
    }

    results["2_exploration"].update(basic_stats(imdb))
    results["2_exploration"].update(plot_votes_distribution(imdb, outdir))
    results["2_exploration"].update(revenue_rating_relation(imdb, outdir))

    results["3_genre_analysis"].update(genre_analysis(imdb, outdir))
    results["4_director_actor"].update(director_actor_analysis(imdb, outdir))
    results["5_year_runtime"].update(year_runtime_analysis(imdb, outdir))
    results["6_clustering"].update(clustering(imdb, outdir))

    ratings = pd.read_csv(rating_path)
    results["7_recommendation"].update(recommendation_analysis(ratings))

    with open(outdir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="电影数据分析与推荐系统作业脚本")
    parser.add_argument("--imdb", type=Path, default=Path("IMDB-Movie-Data.csv"))
    parser.add_argument("--ratings", type=Path, default=Path("movies-rating.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    run(args.imdb, args.ratings, args.outdir)


if __name__ == "__main__":
    main()
