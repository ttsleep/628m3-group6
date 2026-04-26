# 📊 AeroPredict 技术报告
**STAT 628 Module 3 | Group 6 | 内部组员对齐文档**

*Last Updated: 2026-04-23*

---

## 一、 当前进度总览

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据预处理脚本 | ✅ 完成 | 含节日特征工程 |
| 时区 UTC 转换（含红眼修正） | ✅ 完成 | 教授推荐方案 |
| 节日特征工程 | ✅ 完成 | WeekOfMonth + days_to_holiday |
| 机器学习模型训练 | ✅ 完成 | 8 特征版本，已重新训练 |
| Dash Web App | ✅ 完成 | 含日期选择器、节日感知预测 |
| Shiny Web App | ✅ 完成 | 用于 shinyapps.io 公网部署 |
| GitHub 仓库同步 | ✅ 完成 | 最新 commit: 节日特征版本 |
| Executive Summary | ⬜ 待完成 | |
| Technical Report PDF | ⬜ 待完成 | |
| Presentation Slides | ⬜ 待完成 | |
| 公网部署（云端 URL） | ⬜ 待完成 | 优先级：高 |

---

## 二、 数据范围与来源

### 数据来源
美国交通部（U.S. Department of Transportation）按月提供的商业航班准点数据。

### 我们使用的数据范围

| 维度 | 范围 |
|------|------|
| 时间 | 2023年9–12月、2024年9–12月、2025年9–11月（共11个月） |
| 航空公司 | **AA**（美国航空）、**DL**（达美航空）、**UA**（美联航）|
| 机场 | ATL, DFW, DEN, ORD, LAX, JFK, LAS, MCO, MIA, CLT, SEA, PHX, EWR, SFO, IAH（15个枢纽）|
| 最终数据量 | **588,993 条航班记录** |

> **注意**：2025年12月数据尚未公开发布，正是 TA 用于评分的"未公开数据集"。我们的模型基于 9–12 月的历史季节规律进行预测，节日特征可显著提升对12月圣诞节周的预测精度。

---

## 三、 数据预处理详解（`scripts/preprocess_data.py`）

### 3.1 核心挑战：时区多元化
原始数据中所有时间戳都是**当地时间**——起飞时间是以出发机场时区记录的，降落时间是以目的地机场时区记录的。无法直接计算两者之差。

### 3.2 解决方案：统一 UTC
1. 加载 `airports_timezone.csv` 获取每个机场的 IANA 时区（如 `America/New_York`）
2. 将出发地时区映射到每行数据中
3. 使用 Python 的 `pytz` 库，将本地时间强制转换为 UTC 绝对时间

```python
# 转换示例：LAX 出发时间本地时间 → UTC
local_tz = pytz.timezone('America/Los_Angeles')
localized = dt.dt.tz_localize(local_tz, ambiguous='NaT', nonexistent='NaT')
utc_converted = localized.dt.tz_convert('UTC')
```

### 3.3 红眼航班的处理（教授推荐方法）

> **问题**：红眼航班深夜出发、次日凌晨降落。如果以同一天的本地时间转 UTC，会得到"到达早于出发"的错误结果。

> **教授推荐方案（已采用）**：
> ```
> 到达UTC时间 = 出发UTC时间 + 计划飞行时长（CRSElapsedTime，单位：分钟）
> ```
> 这完全绕开了"猜测哪天降落"的问题，是最可靠的方法。

### 3.4 节日特征工程（新增）

这是本次最重要的模型改进之一，专门为捕捉感恩节、圣诞节等节日对航班的影响而设计。

#### 3.4.1 WeekOfMonth（月内第几周）

```python
df['WeekOfMonth'] = (df['DayofMonth'] - 1) // 7 + 1
```

| 值 | 含义 | 典型场景 |
|----|------|---------|
| 1 | 第1周（1–7日） | 普通周 |
| 2 | 第2周（8–14日） | 普通周 |
| 3 | 第3周（15–21日） | 感恩节前旅行高峰 |
| 4 | 第4周（22–28日） | **感恩节当周**（11月）/ **圣诞前周**（12月）|

> 感恩节固定是11月第4个周四（`WeekOfMonth=4, DayOfWeek=4, Month=11`），模型可精确识别。

#### 3.4.2 days_to_nearest_holiday（距最近节日天数）

```python
def get_thanksgiving(year):
    nov1 = datetime.date(year, 11, 1)
    days_to_thu = (3 - nov1.weekday()) % 7
    first_thu = nov1 + timedelta(days=days_to_thu)
    return first_thu + timedelta(weeks=3)   # 4th Thursday

def days_to_nearest_holiday(date):
    holidays = [
        datetime.date(year, 10, 31),   # 万圣节
        get_thanksgiving(year),         # 感恩节（动态计算）
        datetime.date(year, 12, 25),   # 圣诞节
        datetime.date(year, 12, 31),   # 元旦前夜
    ]
    return min(abs((date - h).days) for h in holidays)
```

该特征值越小，说明离节日越近，模型据此提升延误概率预测。圣诞节 `days_to_holiday=0`，圣诞前夕 `days_to_holiday=1`，有效捕捉12月旅行峰值。

### 3.5 输出
生成 `processed_flights_utc.csv`，包含：
- 出发/到达的 UTC 时间戳
- 原始的 `DepDelay`、`ArrDelay`（直接使用原始字段）
- `Cancelled` 标志、`TaxiOut`、`TaxiIn`
- **新增**：`WeekOfMonth`、`days_to_holiday`

---

## 四、 机器学习建模（`scripts/train_models.py`）

### 4.1 特征工程（8 特征版本）

| 特征名 | 类型 | 说明 |
|--------|------|------|
| `Origin_code` | 类别编码 | 出发机场 |
| `Dest_code` | 类别编码 | 目的地机场 |
| `Airline_code` | 类别编码 | 航空公司 |
| `Month` | 数值 | 月份（9–12） |
| `DayOfWeek` | 数值 | 星期几（1=周一） |
| `CRSDepHour` | 数值 | UTC 计划起飞小时（0–23） |
| `WeekOfMonth` ⭐ | 数值 | 月内第几周（1–5）|
| `days_to_holiday` ⭐ | 数值 | 距最近节日天数 |

⭐ = 本次新增特征

> **设计说明**：不直接使用"几号（DayofMonth）"作为特征，而是用 WeekOfMonth + days_to_holiday 的组合，既能捕捉节日周期规律（WeekOfMonth捕捉感恩节），又能捕捉节日临近效应（days_to_holiday捕捉圣诞），同时避免模型对特定日期过拟合。

### 4.2 模型一：取消概率（二分类）

- **算法**：`RandomForestClassifier`（随机森林，50棵树，max_depth=15）
- **输出**：0–100% 的取消概率
- **保存**：`models/clf_cancelled.joblib`

### 4.3 模型二 & 三：延误预测（量化不确定性）

这是项目最关键的技术亮点，直接解决了课设要求"必须附带不确定性（Uncertainties）"的核心难点。

- **算法**：`HistGradientBoostingRegressor`，设置 `loss='quantile'`（分位数损失）
- **训练了 6 个模型**（起飞 × 3 分位 + 到达 × 3 分位）：

| 分位数 | 含义 | 文件 |
|--------|------|------|
| q=0.10 | 延误的乐观下界（10%的情况下结果更好） | |
| q=0.50 | 中位数预期延误（最可能的结果） | |
| q=0.90 | 延误的悲观上界（90%情况下不会超过） | |

这三条线合起来形成了一个 **80% 预测区间**，即我们的"不确定性范围"。

- **保存**：`models/reg_dep_delay.joblib`，`models/reg_arr_delay.joblib`

### 4.4 辅助统计：滑行时间
- 按出发机场统计 `TaxiOut` 历史均值
- 按目的地机场统计 `TaxiIn` 历史均值
- 保存：`models/taxi_stats.joblib`

---

## 五、 Web 应用

### 5.1 双框架策略

| | Dash App（`app/app.py`） | Shiny App（`app/app_shiny.py`）|
|-|--------------------------|-------------------------------|
| 框架 | Plotly Dash + DBC | Shiny for Python + shinyswatch |
| 主题 | DARKLY 深色 | Darkly bootstrap |
| 运行端口 | 8050 | 8051 |
| 用途 | 本地开发 / 演示 | **shinyapps.io 公网部署** |
| 图表渲染 | `dcc.Graph` | `plotly.to_html()` + `ui.HTML()` |

### 5.2 用户输入（新版）

| 控件 | 类型 | 说明 |
|------|------|------|
| Origin Airport | 下拉菜单 | 15 个枢纽机场 |
| Destination Airport | 下拉菜单 | 动态过滤，不能与出发地相同 |
| Airline Carrier | 下拉菜单 | AA / DL / UA |
| **Flight Date** ⭐ | **日历日期选择器** | **2023-09-01 至 2025-12-31；自动推导月份、星期几、WeekOfMonth、days_to_holiday** |
| UTC Hour | 滑块 | 0–23 时 |

> **日期选择器的设计意图**：用户只需选一个具体日期，系统自动提取所有时间维度特征，无需手动填写月份和星期几，且节日特征完全透明地自动计算，对用户无感知。

### 5.3 输入验证（防止越界预测）

```python
if month not in range(9, 13):
    # 显示黄色警告，拒绝给出预测
    warn_msg = "⚠️ 该月份超出训练数据范围（仅支持9–12月）"
    return empty_fig, empty_fig, warn_msg
```

用户如选择 1–8 月的日期，App 会显示警告而非给出无意义的越界预测。

### 5.4 预测输出可视化

**① 取消概率仪表盘（Gauge Chart）**
- 量程 0–5%（对微小概率差异高度敏感）
- 颜色分区：绿（0–1%）、黄（1–2.5%）、红（2.5–5%）

**② 延误区间误差图（Error Bar Plot）**
- 橙色点：起飞延误中位数，带 10th–90th 百分位误差棒
- 紫色点：到达延误中位数，带误差棒
- Y 轴单位：分钟（负数 = 提前）

**③ 滑行时间洞察（文字说明）**
- 出发机场平均 Taxi-Out 分钟数
- 目的地机场平均 Taxi-In 分钟数

---

## 六、 如何在本地运行

```bash
# 克隆仓库
git clone git@github.com:ttsleep/628m3-group6.git
cd 628m3-group6

# 安装依赖
pip install pandas pytz scikit-learn dash dash-bootstrap-components plotly joblib
pip install shiny shinyswatch  # 仅 Shiny App 需要

# 启动 Dash App（端口 8050）
python app/app.py

# 启动 Shiny App（端口 8051）
shiny run app/app_shiny.py --port 8051 --reload

# 如需重新训练（模型已在仓库中，通常无需执行）
python scripts/preprocess_data.py  # ~2 分钟
python scripts/train_models.py     # ~10 分钟
```

---

## 七、 模型局限性说明

在报告和答辩中应主动提及以下局限性，展示严谨的学术态度：

| 局限性 | 说明 |
|--------|------|
| 无实时天气数据 | 无法感知当天的风暴、雷雨等极端天气，这些是航班延误的最大随机因素 |
| 仅覆盖9–12月 | 模型拒绝对1–8月给出预测，输入范围外数据无意义 |
| 仅3家航司15机场 | 未涵盖低成本航空（Southwest等）和小型机场 |
| 节日覆盖有限 | 仅明确建模了万圣节、感恩节、圣诞节、元旦；其他节假日（如感恩节前的周三）通过 WeekOfMonth 间接捕捉 |

---

## 八、 GitHub 仓库与提交历史

🔗 **https://github.com/ttsleep/628m3-group6**

| Commit | 说明 |
|--------|------|
| `f9a55e3` （最新）| ✅ **节日特征（WeekOfMonth + days_to_holiday）+ 重训所有模型** |
| `d547777` | ✅ 中文技术报告 + README 更新 |
| `cd2a32a` | ✅ README 完整结构说明 |
| `ae1fca2` | ✅ 对齐教授红眼航班方案，重训模型 |
| `f038ddb` | ✅ 项目初始提交 |
