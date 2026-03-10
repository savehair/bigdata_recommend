# 电影数据分析与推荐（Python脚本版）

本仓库提供 `movie_analysis_and_recommendation.py`，用于完成题目中的 7 个部分：

1. IMDB 数据读取与预处理
2. 数据探索与统计分析
3. 电影类型分析
4. 导演和演员分析
5. 年份与时长分析
6. KMeans 聚类（5类）+ 轮廓系数
7. 基于 `movies-rating` 的协同过滤推荐（余弦/皮尔逊）

## 运行方式

```bash
python movie_analysis_and_recommendation.py \
  --imdb IMDB-Movie-Data.csv \
  --ratings movies-rating.csv \
  --outdir outputs
```

## 输出内容

- `outputs/analysis_results.json`：所有题目对应的核心统计结果。
- 多张 PNG 可视化图（如评分人数分布、票房与评分关系、年份分布、聚类散点图等）。

## 依赖

建议 Python 3.9+，并安装：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
