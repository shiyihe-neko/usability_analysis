# reading_evaluator/__init__.py
import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_reading_tasks(all_data, format_list=None):
    """
    提取所有 reading-task-...-<format>-<n> 条目，并将 list 答案扁平化为 "A, B" 形式。
    返回 DataFrame，列：
      ['participantId','format','task','startTime','endTime',
       'duration_sec','q','helpButtonClickedCount']
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

        for key, content in answers.items():
            if not isinstance(content, dict):
                continue
            comp = content.get('componentName', key)
            m = re.match(r'^(reading-task-[^-]+-[^-]+)-(\d+)$', comp)
            if not m:
                continue
            base, part = m.group(1), m.group(2)
            fmt = base.split('-')[-1].lower()

            if format_list and fmt not in format_list:
                continue

            st, ed = content.get('startTime'), content.get('endTime')
            dur = (ed - st)/1000.0 if st is not None and ed is not None else None

            ansb = content.get('answer', {}) or {}
            q_val = None
            for k, v in ansb.items():
                if re.search(rf'_q{part}$', k):
                    if isinstance(v, list):
                        q_val = ', '.join(str(x) for x in v)
                    else:
                        q_val = v
                    break

            help_cnt = content.get('helpButtonClickedCount')
            rows.append({
                'participantId': pid,
                'format': fmt,
                'task': comp,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur,
                'q': q_val,
                'helpButtonClickedCount': help_cnt
            })

    df = pd.DataFrame(rows)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    return df


def sanitize_reading_task_names(df_tasks):
    """
    去掉 task 名中的 “-<format>” 部分：
      reading-task-tabular-hjson-1 → reading-task-tabular-1
    """
    df = df_tasks.copy()
    df['task_clean'] = df.apply(
        lambda r: r['task'].replace(f"-{r['format']}", ""), axis=1
    )
    return df


def summarize_reading_times(df_tasks):
    """
    计算：
      1) 各 format 在各 task_clean 上的平均 duration_sec
      2) 各 format 下，每位参与者的总时长平均值
    返回 (df_task_avg, df_part_avg)
    """
    df = sanitize_reading_task_names(df_tasks)

    df_task_avg = (
        df.groupby(['format','task_clean'])['duration_sec']
          .mean()
          .reset_index()
          .rename(columns={'task_clean':'task','duration_sec':'average_duration_sec'})
    )

    df_part_total = (
        df.groupby(['participantId','format'])['duration_sec']
          .sum()
          .reset_index()
          .rename(columns={'duration_sec':'total_duration_sec'})
    )

    df_part_avg = (
        df_part_total.groupby('format')['total_duration_sec']
                     .mean()
                     .reset_index()
                     .rename(columns={'total_duration_sec':'average_total_duration_sec'})
    )

    return df_task_avg, df_part_avg


def _normalize_answer(val):
    """
    规范化一个答案：
      - 转为 str，折叠所有空白为一个空格
      - 去掉首尾空白
      - 去掉外层中括号
    """
    if not isinstance(val, str):
        val = str(val)
    s = re.sub(r'\s+', ' ', val).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1].strip()
    return s


def add_correctness(df_tasks, correct_answers):
    """
    对齐 df_tasks，加列 'correct' (1/0)：
      correct_answers: { task_clean: 答案或[多答案] }
    """
    df = sanitize_reading_task_names(df_tasks)
    def is_corr(r):
        corr = correct_answers.get(r['task_clean'])
        if corr is None: return 0
        vals = corr if isinstance(corr,(list,tuple)) else [corr]
        a = _normalize_answer(r['q'])
        return int(any(a==_normalize_answer(c) for c in vals))
    df['correct'] = df.apply(is_corr, axis=1)
    return df


def summarize_reading_performance(df_tasks, correct_answers):
    """
    返回 (df_task_perf, df_part_avg)：
      df_task_perf: ['format','task_clean','average_duration_sec',
                     'correct_rate','average_helpClicks']
      df_part_avg:   ['format','avg_total_duration_sec',
                     'avg_total_correct_rate','avg_total_helpClicks']
    """
    df = add_correctness(df_tasks, correct_answers)

    df_task_perf = (
        df.groupby(['format','task_clean'])
          .agg(
            average_duration_sec=('duration_sec','mean'),
            correct_rate         =('correct','mean'),
            average_helpClicks   =('helpButtonClickedCount','mean')
          )
          .reset_index()
    )

    df_part = (
        df.groupby(['participantId','format'])
          .agg(
            total_duration_sec=('duration_sec','sum'),
            total_correct_rate=('correct','mean'),
            total_helpClicks  =('helpButtonClickedCount','sum')
          )
          .reset_index()
    )

    df_part_avg = (
        df_part.groupby('format')
               .agg(
                 avg_total_duration_sec=('total_duration_sec','mean'),
                 avg_total_correct_rate=('total_correct_rate','mean'),
                 avg_total_helpClicks=('total_helpClicks','mean')
               )
               .reset_index()
    )

    return df_task_perf, df_part_avg


def plot_reading_task_performance(df_task_perf):
    """
    一图三子图，展示各 task_clean 在不同 format 下的：
      1) 平均时长 2) 正确率 3) 平均 help 点击
    """
    fig = plt.figure(constrained_layout=True, figsize=(10,12))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    # 用关键字参数避免位置参数问题
    pivot_time    = df_task_perf.pivot(index='task_clean',
                                       columns='format',
                                       values='average_duration_sec').fillna(0)
    pivot_correct = df_task_perf.pivot(index='task_clean',
                                       columns='format',
                                       values='correct_rate').fillna(0)
    pivot_help    = df_task_perf.pivot(index='task_clean',
                                       columns='format',
                                       values='average_helpClicks').fillna(0)

    pivot_time.plot(kind='bar', ax=ax1)
    ax1.set_title("Avg Duration by Task and Format")
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis='x', rotation=45)

    pivot_correct.plot(kind='bar', ax=ax2)
    ax2.set_title("Correct Rate by Task and Format")
    ax2.set_ylabel("Proportion Correct")
    ax2.tick_params(axis='x', rotation=45)

    pivot_help.plot(kind='bar', ax=ax3)
    ax3.set_title("Avg HelpButtonClickedCount by Task and Format")
    ax3.set_ylabel("Avg Clicks")
    ax3.tick_params(axis='x', rotation=45)

    return fig



def plot_reading_participant_performance(df_part_avg):
    """
    一图三子图，展示各 format 下，参与者平均:
      1) 总时长 2) 总正确率 3) 总 help 点击
    """
    fig = plt.figure(constrained_layout=True, figsize=(8,12))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    # 平均总时长
    df_part_avg.plot(
        x='format',
        y='avg_total_duration_sec',
        kind='bar',
        ax=ax1,
        legend=False
    )
    ax1.set_title("Avg Total Duration per Participant by Format")
    ax1.set_ylabel("Seconds")
    ax1.tick_params(axis='x', rotation=45)

    # 平均总正确率
    df_part_avg.plot(
        x='format',
        y='avg_total_correct_rate',
        kind='bar',
        ax=ax2,
        legend=False
    )
    ax2.set_title("Avg Total Correct Rate per Participant by Format")
    ax2.set_ylabel("Proportion Correct")
    ax2.tick_params(axis='x', rotation=45)

    # 平均总 help 点击
    df_part_avg.plot(
        x='format',
        y='avg_total_helpClicks',
        kind='bar',
        ax=ax3,
        legend=False
    )
    ax3.set_title("Avg Total HelpButtonClickedCount per Participant by Format")
    ax3.set_ylabel("Clicks")
    ax3.tick_params(axis='x', rotation=45)

    return fig

