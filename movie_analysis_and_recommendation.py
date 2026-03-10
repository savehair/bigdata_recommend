"""
一个偏“大学生课程作业风格”的电影数据分析脚本。

功能覆盖：
1) IMDB 数据读取和预处理
2) 数据探索统计 + 可视化
3) 电影类型分析
4) 导演和演员分析
5) 年份和时长分析
6) KMeans 聚类 + 轮廓系数
7) 电影推荐（余弦相似度、皮尔逊相似度）

运行示例：
python movie_analysis_and_recommendation.py \
  --imdb IMDB-Movie-Data.csv \
  --ratings movies-rating.csv \
  --outdir outputs
"""

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

# 统一图表风格（简单白底网格）
sns.set_theme(style="whitegrid")


def make_output_dir(outdir: Path) -> Path:
    """创建输出文件夹（如果不存在）。"""
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def split_and_explode(series: pd.Series) -> pd.Series:
    """把类似 'Action,Drama' 的列拆成一行一个值，用于统计。"""
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
    """IMDB 数据清洗：类型转换、缺失值处理、去重。"""
    data = df.copy()

    # 列名去掉前后空格，防止后面取列名报错
    data.columns = [c.strip() for c in data.columns]

    # 这些列应该是数字，统一转换；异常值变成 NaN
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
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # 文本列缺失值先填 Unknown，避免 split 等操作崩溃
    text_cols = ["Title", "Genre", "Description", "Director", "Actors"]
    for col in text_cols:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype(str).str.strip()

    # 去重策略：优先按 Title + Year 去重，再做全行去重
    if {"Title", "Year"}.issubset(data.columns):
        data = data.sort_values("Rank", na_position="last").drop_duplicates(
            subset=["Title", "Year"], keep="first"
        )
    data = data.drop_duplicates().reset_index(drop=True)

    return data


def part2_exploration(df: pd.DataFrame, outdir: Path) -> dict:
    """第2部分：数据探索和统计分析。"""
    result = {}

    # (1) 基础统计
    directors = split_and_explode(df["Director"]) if "Director" in df else pd.Series(dtype=str)
    actors = split_and_explode(df["Actors"]) if "Actors" in df else pd.Series(dtype=str)
    genres = split_and_explode(df["Genre"]) if "Genre" in df else pd.Series(dtype=str)

    result["movie_count"] = int(len(df))
    result["director_count"] = int(directors.nunique())
    result["actor_count"] = int(actors.nunique())
    result["genre_count"] = int(genres.nunique())

    # (2) Votes 分布
    votes = df["Votes"].dropna()
    result["votes_mean"] = float(votes.mean())
    result["votes_max"] = float(votes.max())
    result["votes_min"] = float(votes.min())

    plt.figure(figsize=(8, 5))
    sns.histplot(votes, bins=30, kde=True)
    plt.title("Votes Distribution")
    plt.xlabel("Votes")
    plt.tight_layout()
    plt.savefig(outdir / "part2_votes_distribution.png", dpi=150)
    plt.close()

    # (3) Revenue 和 Rating 关系
    sub = df[["Revenue (Millions)", "Rating"]].dropna()
    corr = sub["Revenue (Millions)"].corr(sub["Rating"])
    result["revenue_rating_corr"] = float(corr)

    plt.figure(figsize=(7, 5))
    sns.regplot(data=sub, x="Revenue (Millions)", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Revenue vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "part2_revenue_rating_relation.png", dpi=150)
    plt.close()

    return result


def part3_genre_analysis(df: pd.DataFrame, outdir: Path) -> dict:
    """第3部分：电影类型分析。"""
    result = {}

    # 先把多类型拆开统计数量
    genres_all = split_and_explode(df["Genre"])
    genre_counts = genres_all.value_counts()

    result["top_two_genres"] = genre_counts.head(2).to_dict()

    plt.figure(figsize=(10, 5))
    genre_counts.head(15).plot(kind="bar")
    plt.title("Top 15 Genres by Count")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "part3_genre_count_top15.png", dpi=150)
    plt.close()

    # 按时长分短/中/长
    # 题目要求：短<90；中90~120；长>120
    # 这里用区间 [-inf,90), [90,121), [121,inf)
    temp = df.copy()
    temp["runtime_type"] = pd.cut(
        temp["Runtime (Minutes)"],
        bins=[-np.inf, 90, 121, np.inf],
        labels=["short", "medium", "long"],
        right=False,
    )

    # 拆类型，统计不同时长类别中每个电影类型平均分
    explode_temp = temp.assign(Genre=temp["Genre"].str.split(",")).explode("Genre")
    explode_temp["Genre"] = explode_temp["Genre"].str.strip()
    avg_rating = (
        explode_temp.dropna(subset=["runtime_type", "Rating"])
        .groupby(["runtime_type", "Genre"])["Rating"]
        .mean()
    )

    # 每个时长类别找评分最低类型
    lowest_each_runtime = {}
    if not avg_rating.empty:
        idx = avg_rating.groupby(level=0).idxmin()
        for runtime_type, pair in idx.items():
            lowest_each_runtime[str(runtime_type)] = {
                "genre": pair[1],
                "avg_rating": float(avg_rating[pair]),
            }
    result["lowest_genre_by_runtime_type"] = lowest_each_runtime

    # 不同类型平均票房
    avg_revenue = (
        explode_temp.dropna(subset=["Revenue (Millions)"])
        .groupby("Genre")["Revenue (Millions)"]
        .mean()
        .sort_values(ascending=False)
    )
    result["avg_revenue_by_genre"] = avg_revenue.to_dict()

    plt.figure(figsize=(10, 5))
    avg_revenue.head(15).plot(kind="bar")
    plt.title("Top 15 Genres by Avg Revenue")
    plt.ylabel("Avg Revenue (Millions)")
    plt.tight_layout()
    plt.savefig(outdir / "part3_genre_avg_revenue_top15.png", dpi=150)
    plt.close()

    return result


def part4_director_actor(df: pd.DataFrame, outdir: Path) -> dict:
    """第4部分：导演和演员分析。"""
    result = {}

    director_series = split_and_explode(df["Director"])
    actor_series = split_and_explode(df["Actors"])

    result["most_common_director"] = director_series.value_counts().head(1).to_dict()
    result["most_common_actor"] = actor_series.value_counts().head(1).to_dict()

    # 2016 年最常见
    data_2016 = df[df["Year"] == 2016]
    result["most_common_director_2016"] = split_and_explode(data_2016["Director"]).value_counts().head(1).to_dict()
    result["most_common_actor_2016"] = split_and_explode(data_2016["Actors"]).value_counts().head(1).to_dict()

    # 每部电影演员人数和评分关系
    actor_num = df["Actors"].fillna("").str.split(",").apply(
        lambda x: len([a for a in x if str(a).strip()])
    )
    sub = pd.DataFrame({"actor_count": actor_num, "Rating": df["Rating"]}).dropna()
    corr = sub["actor_count"].corr(sub["Rating"])
    result["actor_count_rating_corr"] = float(corr)

    plt.figure(figsize=(7, 5))
    sns.regplot(data=sub, x="actor_count", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Actor Count vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "part4_actor_count_rating_relation.png", dpi=150)
    plt.close()

    return result


def part5_year_runtime(df: pd.DataFrame, outdir: Path) -> dict:
    """第5部分：年份和电影时长分析。"""
    result = {}

    # 年份分布，找数量最少年份
    year_count = df["Year"].value_counts().sort_index()
    min_year = int(year_count[year_count == year_count.min()].index.min())
    result["min_movie_count_year"] = min_year

    plt.figure(figsize=(10, 5))
    year_count.plot(kind="bar")
    plt.title("Movie Count by Year")
    plt.tight_layout()
    plt.savefig(outdir / "part5_year_count_distribution.png", dpi=150)
    plt.close()

    # 时长分布
    runtime = df["Runtime (Minutes)"].dropna()
    result["runtime_mean"] = float(runtime.mean())
    result["runtime_max"] = float(runtime.max())
    result["runtime_min"] = float(runtime.min())

    plt.figure(figsize=(8, 5))
    sns.histplot(runtime, bins=30, kde=True)
    plt.title("Runtime Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "part5_runtime_distribution.png", dpi=150)
    plt.close()

    # 年份和评分关系
    y_r = df[["Year", "Rating"]].dropna()
    result["year_rating_corr"] = float(y_r["Year"].corr(y_r["Rating"]))

    plt.figure(figsize=(7, 5))
    sns.regplot(data=y_r, x="Year", y="Rating", scatter_kws={"alpha": 0.6})
    plt.title("Year vs Rating")
    plt.tight_layout()
    plt.savefig(outdir / "part5_year_rating_relation.png", dpi=150)
    plt.close()

    return result


def part6_kmeans(df: pd.DataFrame, outdir: Path) -> dict:
    """第6部分：KMeans 聚类（5类）和轮廓系数。"""
    result = {}

    # 选常用数值特征
    features = ["Runtime (Minutes)", "Rating", "Votes", "Revenue (Millions)", "Metascore", "Year"]
    sub = df[features].dropna()

    if len(sub) < 6:
        result["cluster_error"] = "Not enough data rows for 5 clusters"
        return result

    x = StandardScaler().fit_transform(sub)

    km = KMeans(n_clusters=5, random_state=42, n_init=20)
    labels = km.fit_predict(x)

    result["silhouette_score"] = float(silhouette_score(x, labels))
    result["cluster_sizes"] = pd.Series(labels).value_counts().sort_index().to_dict()

    # 可视化：画标准化后前2维，简单展示聚类结果
    vis = pd.DataFrame({"f1": x[:, 0], "f2": x[:, 1], "cluster": labels})
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=vis, x="f1", y="f2", hue="cluster", palette="tab10")
    plt.title("KMeans(5) Clusters")
    plt.tight_layout()
    plt.savefig(outdir / "part6_kmeans_clusters.png", dpi=150)
    plt.close()

    return result


def build_user_item(ratings: pd.DataFrame) -> pd.DataFrame:
    """把评分表转成 user-item 矩阵。"""
    return ratings.pivot_table(index="userId", columns="itemId", values="rating")


def get_similarity(user_item: pd.DataFrame, method: str) -> pd.DataFrame:
    """计算用户-用户相似度矩阵。"""
    if method == "cosine":
        filled = user_item.fillna(0)
        sim = cosine_similarity(filled.values)
        return pd.DataFrame(sim, index=filled.index, columns=filled.index)
    if method == "pearson":
        # 皮尔逊相似度直接做用户相关系数
        return user_item.T.corr(method="pearson").fillna(0)
    raise ValueError("method must be cosine or pearson")


def predict_rating(user_item: pd.DataFrame, sim_df: pd.DataFrame, user: int, item: int, topk: int = 20) -> float:
    """用相似用户加权平均预测评分。"""
    if user not in user_item.index or item not in user_item.columns:
        return float("nan")

    # 找到对该物品打过分的用户
    rated_users = user_item[item].dropna().index

    # 取目标用户和这些用户的相似度
    sim_users = sim_df.loc[user].drop(index=user, errors="ignore")
    sim_users = sim_users.loc[sim_users.index.intersection(rated_users)]

    if sim_users.empty:
        # 如果没有邻居，退化成目标用户均值
        user_mean = user_item.loc[user].mean()
        return float(user_mean) if not np.isnan(user_mean) else float("nan")

    # 取最相似的 topk 个用户
    sim_users = sim_users.sort_values(ascending=False).head(topk)
    neighbor_ratings = user_item.loc[sim_users.index, item]

    # 加权平均
    denom = np.abs(sim_users).sum()
    if denom == 0:
        return float(neighbor_ratings.mean())
    return float(np.dot(sim_users.values, neighbor_ratings.values) / denom)


def rmse_for_user(ratings: pd.DataFrame, user: int, method: str) -> float:
    """对用户已评分条目做 leave-one-out 预测，计算 RMSE。"""
    user_data = ratings[ratings["userId"] == user]
    if len(user_data) < 2:
        return float("nan")

    pred_list, true_list = [], []
    for idx, row in user_data.iterrows():
        # 去掉当前这条真实评分，再预测
        train = ratings.drop(index=idx)
        ui = build_user_item(train)
        sim = get_similarity(ui, method)
        pred = predict_rating(ui, sim, user=user, item=row["itemId"])
        if not np.isnan(pred):
            pred_list.append(pred)
            true_list.append(row["rating"])

    if not pred_list:
        return float("nan")

    return float(np.sqrt(mean_squared_error(true_list, pred_list)))


def recommend_topn(ratings: pd.DataFrame, user: int, method: str, topn: int = 5) -> list:
    """给指定用户推荐未看过电影，返回 topn。"""
    ui = build_user_item(ratings)
    sim = get_similarity(ui, method)

    if user not in ui.index:
        return []

    # 没打过分的就是未看过
    unseen_items = ui.loc[user][ui.loc[user].isna()].index.tolist()

    candidates = []
    for item in unseen_items:
        pred = predict_rating(ui, sim, user=user, item=item)
        if not np.isnan(pred):
            candidates.append((int(item), float(pred)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:topn]


def part7_recommendation(ratings: pd.DataFrame) -> dict:
    """第7部分：推荐系统分析。"""
    result = {}

    # 基础清洗，确保类型正确
    data = ratings.copy()
    data["userId"] = pd.to_numeric(data["userId"], errors="coerce").astype("Int64")
    data["itemId"] = pd.to_numeric(data["itemId"], errors="coerce").astype("Int64")
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data = data.dropna(subset=["userId", "itemId", "rating"]).astype({"userId": int, "itemId": int})

    # (1) 正确导入数据集 -> 用数据条数证明
    result["ratings_rows"] = int(len(data))

    # 准备相似度矩阵
    ui = build_user_item(data)
    sim_cos = get_similarity(ui, "cosine")
    sim_pea = get_similarity(ui, "pearson")

    # (2) 预测用户2对电影7评分
    pred_cos = predict_rating(ui, sim_cos, user=2, item=7)
    pred_pea = predict_rating(ui, sim_pea, user=2, item=7)

    result["predict_user2_item7"] = {
        "cosine": None if np.isnan(pred_cos) else float(pred_cos),
        "pearson": None if np.isnan(pred_pea) else float(pred_pea),
    }

    # (3) 两种方法分别在用户2已评分电影上计算RMSE
    rmse_cos = rmse_for_user(data, user=2, method="cosine")
    rmse_pea = rmse_for_user(data, user=2, method="pearson")

    result["rmse_user2"] = {
        "cosine": None if np.isnan(rmse_cos) else float(rmse_cos),
        "pearson": None if np.isnan(rmse_pea) else float(rmse_pea),
    }

    # 选择 RMSE 更小的方法
    if np.isnan(rmse_pea) or (not np.isnan(rmse_cos) and rmse_cos <= rmse_pea):
        best_method = "cosine"
    else:
        best_method = "pearson"
    result["best_method"] = best_method

    # 推荐用户2未看过电影 top5
    recs = recommend_topn(data, user=2, method=best_method, topn=5)
    result["top5_recommendations_for_user2"] = [
        {"itemId": item, "pred_rating": score} for item, score in recs
    ]

    return result


def run_all(imdb_path: Path, ratings_path: Path, outdir: Path) -> dict:
    """总流程函数：读取数据 -> 分模块计算 -> 保存结果。"""
    outdir = make_output_dir(outdir)

    imdb = pd.read_csv(imdb_path)
    imdb = preprocess_imdb(imdb)

    results = {
        "1_preprocess": {
            "rows_after_preprocess": int(len(imdb))
        },
        "2_exploration": part2_exploration(imdb, outdir),
        "3_genre_analysis": part3_genre_analysis(imdb, outdir),
        "4_director_actor": part4_director_actor(imdb, outdir),
        "5_year_runtime": part5_year_runtime(imdb, outdir),
        "6_clustering": part6_kmeans(imdb, outdir),
    }

    ratings = pd.read_csv(ratings_path)
    results["7_recommendation"] = part7_recommendation(ratings)

    # 输出 json，便于写实验报告时直接引用
    result_file = outdir / "analysis_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n===== 全部分析完成 =====")
    print(f"结果文件: {result_file}")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    return results


def main():
    parser = argparse.ArgumentParser(description="电影数据分析与推荐（课程作业版）")
    parser.add_argument("--imdb", type=Path, default=Path("IMDB-Movie-Data.csv"), help="IMDB 电影数据CSV路径")
    parser.add_argument("--ratings", type=Path, default=Path("movies-rating.csv"), help="用户评分数据CSV路径")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="输出目录")
    args = parser.parse_args()

    run_all(args.imdb, args.ratings, args.outdir)


if __name__ == "__main__":
    main()
