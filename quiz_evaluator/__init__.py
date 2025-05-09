import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
import math
from scipy.stats import levene, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Tuple

def evaluate_quiz_answers_from_tutorial(all_data):
    """
    Iterate through all_data to evaluate tutorial quiz answers:
      - Extract participantId from prolificId or filename
      - Extract format from tutorial-<format>-part1 key
      - For each tutorial part (part1/part2), compare user answers against correct answers
      - Count wrong attempts and distributions
      - Return a DataFrame of quiz results per participant, quiz key, and format
    """
    quiz_results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # 1. 提取 participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                answer_block = task_info.get('answer', {})
                if isinstance(answer_block, dict) and 'prolificId' in answer_block:
                    participant_id = answer_block['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # 2. 提取 format
        format_name = None
        for k in answers.keys():
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break
        format_name = format_name or "unknown"

        # 3. 遍历每个 quiz 任务
        for task_key, task_info in answers.items():
            if not isinstance(task_info, dict):
                continue
            if not re.match(r"tutorial-\w+-part[12]$", task_key):
                continue

            # 3.1 拿到正确答案
            correct_ans_list = task_info.get("correctAnswer", [])
            if not correct_ans_list or not isinstance(correct_ans_list[0], dict):
                continue
            quiz_id = correct_ans_list[0].get("id")
            correct_answer = correct_ans_list[0].get("answer", [])
            correct_set = set(correct_answer)

            # 3.2 拿到用户最终答案
            answer_block = task_info.get("answer", {})
            user_final_ans = answer_block.get(quiz_id, [])
            is_correct = set(user_final_ans) == correct_set

            # 3.3 汇总所有错误尝试（来自 incorrectAnswers → value 字段）
            incorrect_info = task_info.get("incorrectAnswers", {}).get(quiz_id, {})
            # `value` 是一个 list of lists，例如 [['B','C'], ['B']]
            attempts = incorrect_info.get("value", [])

            # 3.4 统计每个选项出现频次
            counter = Counter()
            for attempt in attempts:
                counter.update(attempt)

            # 跳过正确选项，留下纯错误分布
            wrong_choice_distribution = {
                choice: cnt for choice, cnt in counter.items()
                if choice not in correct_set
            }
            wrong_choice_count = sum(wrong_choice_distribution.values())

            quiz_results.append({
                "participantId": participant_id,
                "format":       format_name,
                "quiz_key":     task_key,
                "correct_answer":        correct_answer,
                "user_final_answer":     user_final_ans,
                "is_correct":            is_correct,
                "num_wrong_attempts":    len(attempts),
                "all_wrong_attempts_list":      attempts,
                "all_wrong_attempts_frequency": dict(counter),
                "wrong_choice_distribution":    wrong_choice_distribution,
                "wrong_choice_count":           wrong_choice_count
            })

    return pd.DataFrame(quiz_results)

def _aggregate_dicts(series):
    """
    把一列 dict 累加后取平均。
    """
    total = Counter()
    n = len(series)
    for d in series:
        total.update(d)
    return {k: v / n for k, v in total.items()}


def plot_quiz_metrics_by_group(df_quiz,
                               group_by='format',
                               metrics=None,
                               format_list=None):
    """
    在单个 Figure 里，可视化按 group_by 分组后的各项指标。

    参数：
      df_quiz       DataFrame，evaluate_quiz_answers_from_tutorial 的输出
      group_by      'format' 或 'quiz_key'
      metrics       要绘制的指标列表，可选：
                     'correct_rate', 'num_wrong_attempts',
                     'wrong_choice_count',
                     'all_wrong_attempts_frequency',
                     'wrong_choice_distribution'
                   默认全量五项。
      format_list   list[str] 或 None：在任何 group_by 模式下，
                   都先用它来过滤 df_quiz['format']。

    特别说明：
      - 如果 group_by=='format'，会自动去掉
        'wrong_choice_distribution' 和 'all_wrong_attempts_frequency'，
        因为你不想在按 format 时看到这两项。
      - format_list 始终生效，无论 group_by。
    """
    # —— 1. 根据 format_list 先过滤
    if format_list is not None:
        df_quiz = df_quiz[df_quiz['format'].isin(format_list)]

    # —— 2. 校验 group_by
    if group_by not in ('format', 'quiz_key'):
        raise ValueError("group_by must be 'format' or 'quiz_key'")

    # —— 3. 默认指标
    if metrics is None:
        metrics = [
            'correct_rate',
            'num_wrong_attempts',
            'wrong_choice_count',
            'all_wrong_attempts_frequency',
            'wrong_choice_distribution'
        ]

    # —— 4. 如果是按 format 画图，就去掉那两项
    if group_by == 'format':
        metrics = [
            m for m in metrics
            if m not in ('all_wrong_attempts_frequency', 'wrong_choice_distribution')
        ]

    # —— 5. 数值指标对应列名
    numeric_map = {
        'correct_rate':       'is_correct',
        'num_wrong_attempts': 'num_wrong_attempts',
        'wrong_choice_count': 'wrong_choice_count'
    }

    # —— 6. 准备画布
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    # —— 7. 逐指标绘制
    for ax, metric in zip(axes, metrics):
        if metric in numeric_map:
            col = numeric_map[metric]
            summary = df_quiz.groupby(group_by)[col].mean()
            ax.bar(summary.index, summary.values)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by {group_by}")

        elif metric == 'all_wrong_attempts_frequency':
            freq_series = df_quiz.groupby(group_by)[metric].apply(_aggregate_dicts)
            df_freq = pd.DataFrame(
                list(freq_series.values),
                index=freq_series.index
            ).fillna(0)
            df_freq.plot(kind='bar', ax=ax, legend=True)
            ax.set_ylabel("avg freq")
            ax.set_title(f"{metric} by {group_by}")

        elif metric == 'wrong_choice_distribution':
            dist_series = df_quiz.groupby(group_by)[metric].apply(_aggregate_dicts)
            df_dist = pd.DataFrame(
                list(dist_series.values),
                index=dist_series.index
            ).fillna(0)
            df_dist.plot(kind='bar', ax=ax, legend=True)
            ax.set_ylabel("avg count")
            ax.set_title(f"{metric} by {group_by}")

        else:
            raise ValueError(f"Unknown metric: {metric}")

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig
    

def process_quiz_metrics_per_participant(
    df_quiz: pd.DataFrame,
    format_list: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Aggregate quiz data to one row per participant × format, computing:
         - correct_rate       = mean of is_correct
         - num_wrong_attempts = mean of num_wrong_attempts
         - wrong_choice_count = mean of wrong_choice_count
    2) Compute the format-level means of those three metrics.

    Args:
      df_quiz     Original quiz-level DataFrame.
      format_list Optional list of formats to keep.

    Returns:
      df_part_summary      DataFrame with columns
        ['participantId','format','correct_rate','num_wrong_attempts','wrong_choice_count']
      df_format_mean       DataFrame with columns
        ['format','correct_rate','num_wrong_attempts','wrong_choice_count']
        where each row is the average across participants for that format.
    """
    df = df_quiz.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]

    # Participant-level summary
    df_part_summary = (
        df
        .groupby(['participantId','format'], as_index=False)
        .agg(
            correct_rate       = ('is_correct',        'mean'),
            num_wrong_attempts = ('num_wrong_attempts','mean'),
            wrong_choice_count = ('wrong_choice_count','mean'),
        )
    )

    # Format-level mean summary
    df_format_mean = (
        df_part_summary
        .groupby('format', as_index=False)
        .agg(
            correct_rate       = ('correct_rate',       'mean'),
            num_wrong_attempts = ('num_wrong_attempts', 'mean'),
            wrong_choice_count = ('wrong_choice_count', 'mean'),
        )
    )

    return df_part_summary, df_format_mean

def test_quiz_metrics(
    df_summary: pd.DataFrame,
    metrics: list = None,
    group_col: str = 'format',
    alpha: float = 0.05
) -> dict:
    """
    For each metric in `metrics`, test whether participant-level values
    differ by `group_col` using:
      1. Levene’s test for equal variances
      2. One-way ANOVA if p_levene > alpha; otherwise Kruskal–Wallis
      3. Tukey HSD post-hoc if ANOVA chosen and p < alpha

    Builds an English interpretation for each metric.

    Returns a dict mapping metric -> {
      'levene': (W, p_levene),
      'method': 'ANOVA' or 'Kruskal–Wallis',
      'stat': F or H statistic,
      'p_value': p-value,
      'tukey': TukeyHSDResults or None,
      'interpretation': str
    }
    """
    # Default metrics
    if metrics is None:
        metrics = ['correct_rate','num_wrong_attempts','wrong_choice_count']

    results = {}
    for metric in metrics:
        if metric not in df_summary.columns:
            print(f"⚠️ Column '{metric}' not found in summary; skipping.")
            continue

        sub = df_summary[[group_col, metric]].dropna()
        groups = [g[metric].values for _, g in sub.groupby(group_col)]
        if len(groups) < 2:
            print(f"⚠️ Not enough formats to compare for '{metric}'; skipping.")
            continue

        # 1) Levene’s test
        w_stat, p_levene = levene(*groups)

        # 2) Choose ANOVA vs Kruskal–Wallis
        if p_levene > alpha:
            stat, pval = f_oneway(*groups)
            method = 'ANOVA'
        else:
            stat, pval = kruskal(*groups)
            method = 'Kruskal–Wallis'

        # 3) Tukey HSD if ANOVA & significant
        tukey = None
        if method == 'ANOVA' and pval < alpha:
            tukey = pairwise_tukeyhsd(
                endog=sub[metric],
                groups=sub[group_col],
                alpha=alpha
            )

        # 4) Build interpretation
        var_msg = (
            f"Levene’s test p = {p_levene:.3f} "
            f"({'homogeneous' if p_levene>alpha else 'heterogeneous'}) variances."
        )
        if pval < alpha:
            main_msg = (
                f"{method} p = {pval:.3f} (< {alpha}): "
                "formats differ significantly."
            )
            if tukey is not None:
                main_msg += " See Tukey HSD for pairwise contrasts."
        else:
            main_msg = (
                f"{method} p = {pval:.3f} (≥ {alpha}): "
                "no significant format differences."
            )

        interp = f"Metric '{metric}': {var_msg} {main_msg}"

        results[metric] = {
            'levene': (w_stat, p_levene),
            'method': method,
            'stat': stat,
            'p_value': pval,
            'tukey': tukey,
            'interpretation': interp
        }

    return results



def analyze_nasa_and_post_surveys(all_data):
    """
    Analyze NASA-TLX and post-task surveys:
      - Extract participantId and format from tutorial keys
      - Build two DataFrames: df_nasa (NASA-TLX) and df_post (post-task survey)
    Returns:
      df_nasa: columns=['participantId','format','startTime','endTime','duration_sec',
                        'mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']
      df_post: columns=[…post-task survey fields…]
    """
    nasa_rows = []
    post_survey_rows = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break

        # format
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # NASA-TLX
        key = '$nasa-tlx.co.nasa-tlx'
        if key in answers:
            info = answers[key]
            ans = info.get('answer', {})
            st, ed = info.get('startTime'), info.get('endTime')
            dur = (ed-st)/1000.0 if st and ed else None
            row = {
                'participantId': pid,
                'format': fmt,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur
            }
            for dim in ['mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']:
                row[dim] = ans.get(dim)
            nasa_rows.append(row)

    df_nasa = pd.DataFrame(nasa_rows)
    return df_nasa

def plot_nasa_tlx_by_format(df_nasa,
                            format_list=None,
                            metrics=None):
    """
    按 format 在一张图里可视化平均 duration_sec 和各 NASA-TLX 维度。

    参数：
      df_nasa     pandas.DataFrame，
                  包含列 ['participantId','format','duration_sec',
                  'mental-demand','physical-demand','temporal-demand',
                  'performance','effort','frustration']
      format_list list[str] or None：只绘制这些 format；None 则绘制全部
      metrics     list[str] or None：要绘制的列，默认：
                  ['duration_sec','mental-demand','physical-demand',
                   'temporal-demand','performance','effort','frustration']
    返回：
      matplotlib.figure.Figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 过滤指定格式
    df = df_nasa.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]

    # 默认要绘制的度量
    if metrics is None:
        metrics = [
            'duration_sec',
            'mental-demand',
            'physical-demand',
            'temporal-demand',
            'performance',
            'effort',
            'frustration'
        ]

    # 强制转为数值，非数值置 NaN
    df[metrics] = df[metrics].apply(pd.to_numeric, errors='coerce')

    # 准备画布
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    # 分组并绘制
    for ax, metric in zip(axes, metrics):
        summary = df.groupby('format')[metric].mean()
        ax.bar(summary.index, summary.values)
        ax.set_title(f"Average {metric} by format")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def test_nasa_metrics_by_format(
    df: pd.DataFrame,
    metrics: list = None,
    format_list: list = None,
    group_col: str = 'format',
    alpha: float = 0.05
) -> dict:
    """
    For each NASA‐TLX metric in `metrics`, test whether values differ across
    formats, guarding against non-numeric entries.

    Steps:
      1. Optionally filter to a subset of formats.
      2. Coerce each metric column to numeric, drop non‐numeric/NaN.
      3. Perform Levene’s test for variance homogeneity.
      4. If p_levene > alpha: one‐way ANOVA; else Kruskal–Wallis.
      5. If ANOVA and p < alpha: Tukey HSD post‐hoc.
      6. Build an English interpretation string.

    Returns a dict mapping metric -> { … }
    """
    # 1) 过滤 formats（如有）
    df_work = df.copy()
    if format_list is not None:
        df_work = df_work[df_work[group_col].isin(format_list)]

    # 2) 默认指标列表
    if metrics is None:
        metrics = [
            'mental-demand',
            'physical-demand',
            'temporal-demand',
            'performance',
            'effort',
            'frustration'
        ]

    results = {}
    for metric in metrics:
        if metric not in df_work.columns:
            print(f"⚠️ Column '{metric}' not found; skipping.")
            continue

        # 强制转换为数值型，错误的条目变 NaN
        df_work[metric] = pd.to_numeric(df_work[metric], errors='coerce')

        # 2b) 丢弃 NaN
        sub = df_work[[group_col, metric]].dropna()
        groups = [g[metric].values for _, g in sub.groupby(group_col)]
        if len(groups) < 2:
            print(f"⚠️ Not enough formats for metric '{metric}'; skipping.")
            continue

        # 3) Levene’s test
        w_stat, p_levene = levene(*groups)

        # 4) 选择 ANOVA 或 Kruskal–Wallis
        if p_levene > alpha:
            stat, pval = f_oneway(*groups)
            method = 'ANOVA'
        else:
            stat, pval = kruskal(*groups)
            method = 'Kruskal–Wallis'

        # 5) Tukey HSD
        tukey = None
        if method == 'ANOVA' and pval < alpha:
            tukey = pairwise_tukeyhsd(
                endog=sub[metric],
                groups=sub[group_col],
                alpha=alpha
            )

        # 6) 生成解读
        var_msg = (
            f"Levene’s test p = {p_levene:.3f} "
            f"({'homogeneous' if p_levene>alpha else 'heterogeneous'}) variances."
        )
        if pval < alpha:
            main_msg = (
                f"{method} p = {pval:.3f} (< {alpha}): "
                "significant differences among formats."
            )
            if tukey is not None:
                main_msg += " See Tukey HSD for pairwise comparisons."
        else:
            main_msg = (
                f"{method} p = {pval:.3f} (≥ {alpha}): "
                "no significant differences among formats."
            )

        interp = f"Metric '{metric}': {var_msg} {main_msg}"

        results[metric] = {
            'levene': (w_stat, p_levene),
            'method': method,
            'stat': stat,
            'p_value': pval,
            'tukey': tukey,
            'interpretation': interp
        }

    return results


def extract_post_task_questions(all_data):
    """
    Extract post-task-question responses into a DataFrame with columns:
      ['participantId','format','task','startTime','endTime','duration_sec',
       'difficulty','confidence']
    Ensures difficulty & confidence are numeric.
    """
    rows = []
    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # participantId
        pid = file_name
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break

        # format
        fmt = "unknown"
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break

        # post-task-question
        for key, content in answers.items():
            if not key.endswith('_post-task-question'): 
                continue
            if not isinstance(content, dict):
                continue

            task = key.replace('_post-task-question', '')
            ans = content.get('answer', {})
            start, end = content.get('startTime'), content.get('endTime')
            duration = (end - start)/1000.0 if start and end else None
            diff = ans.get('difficulty')
            conf = ans.get('confidence')

            rows.append({
                'participantId': pid,
                'format': fmt,
                'task': task,
                'startTime': start,
                'endTime': end,
                'duration_sec': duration,
                'difficulty': diff,
                'confidence': conf
            })

    df = pd.DataFrame(rows)
    # 强制转数值，否则 groupby.mean 会报字符串拼接的错
    for col in ['duration_sec','difficulty','confidence']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_post_task_questions_by_format(df_post, format_list=None):
    """
    Visualize post-task-question DataFrame by format.
      - Numeric columns → average per format
      - String columns  → count of each option per format
    Params:
      df_post      DataFrame from extract_post_task_questions
      format_list  list[str] or None: 只绘制这些 format；None 则绘制全部
    Returns:
      matplotlib.figure.Figure
    """
    # 1. 过滤 format 子集
    df = df_post.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]

    # 2. 确定要绘制的列（排除元信息）
    meta = {'participantId','format','startTime','endTime','task'}
    metrics = [c for c in df.columns if c not in meta]

    # 3. 画布
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    # 4. 逐列绘制
    for ax, col in zip(axes, metrics):
        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值：绘制平均值
            summary = df.groupby('format')[col].mean()
            ax.bar(summary.index, summary.values)
            ax.set_ylabel(f"Avg {col}")
            ax.set_title(f"Average {col} by format")
        else:
            # 字符串：绘制各选项出现频次
            counts = df.groupby(['format', col]).size().unstack(fill_value=0)
            counts.plot(kind='bar', ax=ax)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col} by format")

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def sanitize_task_names(df,
                        format_col='format',
                        task_col='task'):
    """
    Remove the '-<format>' substring from task names.
    E.g. 'reading-task-tabular-hjson-1' → 'reading-task-tabular-1'
    """
    df = df.copy()
    def clean(row):
        fmt  = row[format_col]
        task = row[task_col]
        return task.replace(f"-{fmt}", "")
    df[task_col] = df.apply(clean, axis=1)
    return df

def plot_metrics_by_format_and_task(df_post_questions,
                                    format_list=None,
                                    metrics=None):
    """
    在同一张 Figure 的多个子图里，可视化按 format 和 task 分组后的数值指标平均值。

    参数:
      df_post_questions  DataFrame, 来自 extract_post_task_questions，必须包括列
                         ['format','task'] + metrics 中指定的数值列。
      format_list        list[str] or None: 只保留这些 format；None 则全部。
      metrics            list[str] or None: 要绘制的列，默认 ['duration_sec','difficulty','confidence']。

    返回:
      matplotlib.figure.Figure
    """
    # 默认指标
    if metrics is None:
        metrics = ['duration_sec','difficulty','confidence']

    # 1) 过滤
    df = df_post_questions.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]

    # 2) 清理 task
    df = sanitize_task_names(df)

    # 3) 强制转数值
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4) 分组求平均
    summary = df.groupby(['task','format'])[metrics].mean()

    # 5) 绘图：每个 metric 一个子图
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # 把 format 变成列，task 变成行
        df_m = summary[metric].unstack(level='format').fillna(0)
        df_m.plot(kind='bar', ax=ax)
        ax.set_ylabel(f"Avg {metric}")
        ax.set_title(f"Average {metric} by task and format")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def process_clean_post_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns
      ['participantId','format','task','startTime','endTime',
       'duration_sec','difficulty','confidence',...],
    this function:

      1. Keeps all original rows unchanged.
      2. For each participantId (and format), aggregates any tasks whose names
         start with 'reading-task-tabular-' or 'modifying-task-tabular-'
         into a new row with task_group 'reading-task-tabular' or
         'modifying-task-tabular' respectively.
      3. Aggregation rules:
         - duration_sec: summed across suffix variants
         - difficulty, confidence: averaged across suffix variants
      4. Returns a DataFrame with an extra column 'task_group', containing
         both original task names and the new aggregated group names.
    """
    df = df.copy()
    # keep original task names in a new column task_group
    df['task_group'] = df['task']
    
    agg_frames = []
    # define prefixes to aggregate
    prefixes = ['reading-task-tabular', 'modifying-task-tabular']
    
    for prefix in prefixes:
        # match tasks like 'prefix-1', 'prefix-2', etc.
        mask = df['task'].str.match(rf'^{prefix}-\d+$')
        if not mask.any():
            continue
        
        # select those rows
        tmp = df[mask].copy()
        # group by participantId, format
        grouped = tmp.groupby(['participantId','format'], as_index=False)
        # sum duration, mean difficulty & confidence
        summary = grouped.agg({
            'duration_sec':  'sum',
            'difficulty':    'mean',
            'confidence':    'mean'
        })
        summary['task_group'] = prefix
        # keep only the needed columns plus task_group
        agg_frames.append(summary[[
            'participantId','format','task_group',
            'duration_sec','difficulty','confidence'
        ]])
    
    # combine original and aggregated rows
    df_combined = pd.concat(
        [df[['participantId','format','task','task_group',
             'duration_sec','difficulty','confidence']]] + agg_frames,
        ignore_index=True,
        sort=False
    )
    return df_combined


def test_post_task_metrics(
    df: pd.DataFrame,
    metrics: list = None,
    task_groups: list = None,
    alpha: float = 0.05
) -> dict:
    """
    Within each task_group, test whether the formats differ on each metric.

    Args:
      df            DataFrame must contain ['task_group','format'] + metrics
      metrics       list of metric columns to test; defaults to
                    ['duration_sec','difficulty','confidence']
      task_groups   list of task_group values to include; by default all unique ones
      alpha         significance level

    Returns:
      results: dict keyed by task_group, each is another dict keyed by metric:
        {
          'levene': (W, p_levene),
          'method': 'ANOVA' or 'Kruskal–Wallis',
          'stat': F or H value,
          'p_value': p,
          'tukey': TukeyHSDResults or None,
          'interpretation': str
        }
    """
    if metrics is None:
        metrics = ['duration_sec','difficulty','confidence']
    # decide which task_groups to iterate
    if task_groups is None:
        task_groups = df['task_group'].unique().tolist()
    
    results = {}
    for tg in task_groups:
        df_t = df[df['task_group'] == tg]
        # find formats present
        formats = df_t['format'].dropna().unique().tolist()
        if len(formats) < 2:
            continue
        
        results[tg] = {}
        for metric in metrics:
            if metric not in df_t.columns:
                continue
            # drop missing
            sub = df_t[['format', metric]].dropna()
            groups = [g[metric].values for _, g in sub.groupby('format')]
            if len(groups) < 2:
                continue
            
            # Levene’s test
            w_stat, p_levene = levene(*groups)
            # choose test
            if p_levene > alpha:
                stat, pval = f_oneway(*groups)
                method = 'ANOVA'
            else:
                stat, pval = kruskal(*groups)
                method = 'Kruskal–Wallis'
            # Tukey HSD if ANOVA + significant
            tukey = None
            if method == 'ANOVA' and pval < alpha:
                tukey = pairwise_tukeyhsd(
                    endog=sub[metric],
                    groups=sub['format'],
                    alpha=alpha
                )
            # interpretation
            if p_levene > alpha:
                var_interp = f"Levene’s p = {p_levene:.3f} (> {alpha}), variances homogeneous."
            else:
                var_interp = f"Levene’s p = {p_levene:.3f} (≤ {alpha}), variances heterogeneous."
            if pval < alpha:
                main_interp = (
                    f"{method} p = {pval:.3f} (< {alpha}): "
                    f"formats differ significantly on {metric}."
                )
                if tukey is not None:
                    main_interp += " See Tukey HSD for pairwise."
            else:
                main_interp = (
                    f"{method} p = {pval:.3f} (≥ {alpha}): "
                    f"no significant format differences on {metric}."
                )
            interp = f"[{tg}] {var_interp} {main_interp}"
            
            results[tg][metric] = {
                'levene': (w_stat, p_levene),
                'method': method,
                'stat': stat,
                'p_value': pval,
                'tukey': tukey,
                'interpretation': interp
            }
    return results



import pandas as pd
import re

# 默认熟悉度映射（可按需调整）
_DEFAULT_FAMILIARITY_MAP = {
    "Expert": 4,
    "Comfortable using it": 3,
    "Used it a few times": 2,
    "Heard of it but never used": 1,
    "Not familiar at all": 0
}


def extract_post_task_survey(all_data):
    """
    扁平化提取 post-task-survey-tlx：
      - participantId, format, task, startTime, endTime, duration_sec
      - q7, q8, q9, q9-other, q10, q11
      - q13, q13-other, q14, q14-other
      - q12_<subformat> (子格式熟悉度)
    非字符串答案（如列表）会被拉平为单个字符串。
    """
    rows = []
    for fn, quiz in all_data.items():
        answers = quiz.get('answers', {})

        # participantId
        pid = fn
        for info in answers.values():
            if isinstance(info, dict):
                ab = info.get('answer', {})
                if isinstance(ab, dict) and 'prolificId' in ab:
                    pid = ab['prolificId']
                    break

        # format
        fmt = 'unknown'
        for k in answers:
            m = re.match(r'tutorial-(\w+)-part1', k)
            if m:
                fmt = m.group(1).lower()
                break

        # 定位 post-task-survey-tlx
        key = 'post-task-survey-tlx'
        if key not in answers or not isinstance(answers[key], dict):
            continue
        info = answers[key]

        st, ed = info.get('startTime'), info.get('endTime')
        dur = (ed - st) / 1000.0 if st is not None and ed is not None else None

        ans = info.get('answer', {})
        if not isinstance(ans, dict):
            continue

        row = {
            'participantId': pid,
            'format':        fmt,
            'task':          key,
            'startTime':     st,
            'endTime':       ed,
            'duration_sec':  dur
        }

        # 扁平化 q7–q11, q13–q14
        for q in ['q7','q8','q9','q9-other','q10','q11',
                  'q13','q13-other','q14','q14-other']:
            v = ans.get(q)
            # 如果是列表，拉平成字符串
            if isinstance(v, list):
                if len(v) == 1:
                    flat = v[0]
                else:
                    flat = ';'.join(str(x) for x in v)
            else:
                flat = v
            row[q] = flat

        # 扁平化 q12 子格式熟悉度
        q12 = ans.get('q12', {})
        if isinstance(q12, dict):
            for subfmt, lvl in q12.items():
                row[f"q12_{subfmt.lower()}"] = lvl

        rows.append(row)

    df = pd.DataFrame(rows)
    # 强制数值化 duration_sec 和 q10
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    if 'q10' in df.columns:
        df['q10'] = pd.to_numeric(df['q10'], errors='coerce')
    return df

def plot_post_task_survey_by_format(df_survey,
                                    format_list=None,
                                    familiarity_map=None):
    """
    分格式可视化 post-task-survey-tlx 调查结果，并将每个分类题单独成图：
      - numeric: duration_sec, q10 → 各 format 平均值（单图，多子图）
      - categorical: 每个问题一张图，频次分布（颜色即选项，带图例）
      - q12: 子格式熟悉度 → 平均熟悉度（单图，多子图）
    返回：
      {
        'numeric': Figure,
        'categorical': { question_name: Figure, … },
        'q12': Figure
      }
    """
    # 1. 默认映射
    if familiarity_map is None:
        familiarity_map = _DEFAULT_FAMILIARITY_MAP

    # 2. 过滤 format
    df = df_survey.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]

    # 3. 列分类
    numeric = ['duration_sec']
    if 'q10' in df.columns:
        numeric.append('q10')

    categorical = [
        c for c in df.columns
        if c.startswith('q')
           and c not in numeric
           and not c.startswith('q12_')
    ]
    categorical.sort()

    # 4. 打印映射
    print("\n=== Categorical mappings (excluding q12) ===")
    for col in categorical:
        vals = sorted(df[col].dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(vals)}
        print(f"\n{col}:")
        for v, i in mapping.items():
            print(f"  {v!r} → {i}")

    figs = {}

    # —— 数值题图
    if numeric:
        fig_num, axes_num = plt.subplots(
            len(numeric), 1,
            figsize=(8, 4 * len(numeric)),
            constrained_layout=True
        )
        axes_num = axes_num if len(numeric) > 1 else [axes_num]
        for ax, col in zip(axes_num, numeric):
            summary = df.groupby('format')[col].mean()
            summary.plot(kind='bar', ax=ax, legend=False)
            ax.set_title(f"Avg {col} by format")
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
        figs['numeric'] = fig_num

    # —— 每个分类题单独一张图
    cat_figs = {}
    for col in categorical:
        fig, ax = plt.subplots(
            1, 1,
            figsize=(8, 6),
            constrained_layout=True
        )
        counts = df.groupby(['format', col]).size().unstack(fill_value=0)
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f"{col} distribution by format")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title=col)  # 颜色对应选项
        cat_figs[col] = fig
    figs['categorical'] = cat_figs

    # —— q12 熟悉度图
    q12_cols = sorted(c for c in df.columns if c.startswith('q12_'))
    if q12_cols:
        df_q12 = df[q12_cols].applymap(lambda v: familiarity_map.get(v, pd.NA))
        df_q12['format'] = df['format']
        summary = df_q12.groupby('format')[q12_cols].mean()
        fig_q12, ax_q12 = plt.subplots(
            1, 1,
            figsize=(10, 6),
            constrained_layout=True
        )
        summary.plot(kind='bar', ax=ax_q12)
        ax_q12.set_title("q12 Familiarity by sub-format and format")
        ax_q12.set_ylabel("Avg familiarity")
        ax_q12.tick_params(axis='x', rotation=45)
        ax_q12.legend(title='sub-format')
        figs['q12'] = fig_q12

    return figs


def get_categorical_mappings(df_post_survey):
    """
    自动为所有非数值、以 'q' 开头但不以 'q12_' 开头的列生成 {category: code} 映射。
    返回 dict: { column_name: {category: code, ...}, ... }
    """
    mappings = {}
    for col in df_post_survey.columns:
        # 1) 必须以 'q' 开头
        if not col.startswith('q'):
            continue
        # 2) 排除 q12_ 系列
        if col.startswith('q12_'):
            continue
        # 3) 如果列是数值型，就跳过
        if pd.api.types.is_numeric_dtype(df_post_survey[col]):
            continue

        # 4) 剩下的都是需要映射的分类列
        uniques = df_post_survey[col].dropna().unique().tolist()
        uniques.sort()  # 按字母排序
        mappings[col] = {cat: idx for idx, cat in enumerate(uniques)}

    return mappings


def plot_categorical_mappings(df_post_survey):
    """
    将 get_categorical_mappings 的结果逐列可视化：
      - 每个问题一张水平条形图，Y 轴是类别，X 轴是编码值
    返回 (fig, mappings_dict)
    """
    mappings = get_categorical_mappings(df_post_survey)
    n = len(mappings)
    if n == 0:
        raise ValueError("No categorical q-columns found (besides q12).")

    fig, axes = plt.subplots(
        n, 1,
        figsize=(8, 2.5 * n),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for ax, (col, mp) in zip(axes, mappings.items()):
        cats = list(mp.keys())
        codes = list(mp.values())
        ax.barh(cats, codes)
        ax.set_title(f"Mapping for {col}")
        ax.set_xlabel("Code")
        ax.tick_params(axis='y', rotation=0)

    return fig, mappings