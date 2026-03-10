# 电影数据分析与推荐（课程作业版）

这是一个偏“大学生作业风格”的完整实现，代码里加入了较详细中文注释，便于老师检查和同学阅读。

## 已覆盖内容

- 数据读取和预处理（缺失值、类型转换、去重）
- 数据探索统计（电影数、导演数、演员数等）
- Votes 分布统计与可视化
- Revenue 与 Rating 关系分析与可视化
- 电影类型数量分布、时长分类下的类型评分分析、类型票房分析
- 导演/演员频次分析（含 2016 年）
- 年份分布、时长分布、年份与评分关系
- KMeans 聚类为 5 类 + 轮廓系数
- 基于 movies-rating 的用户协同过滤推荐（余弦/皮尔逊）

## 运行命令

```bash
python movie_analysis_and_recommendation.py
```

运行后会提示输入路径，也可以直接回车使用默认值：
- `IMDB-Movie-Data.csv`
- `movies-rating.pkl`
- `outputs`

评分数据读取支持 `movies-rating.pkl`（推荐）和 `movies-rating.csv`。

## 输出结果

- `outputs/analysis_results.json`：各题的统计结果和推荐结果
- `outputs/*.png`：对应题目的可视化图（含第三部分新增评分图 `part3_avg_rating_by_genre_runtime.png`、`part4_actor_count_rating_relation.png`）

## 依赖安装

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
