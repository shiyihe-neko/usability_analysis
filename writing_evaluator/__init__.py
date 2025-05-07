# writing_evaluator/__init__.py

import re
import pandas as pd
import matplotlib.pyplot as plt
import difflib

def extract_writing_tasks(all_data):
    """
    同前：提取 writing-task-NL，并附带 participantId、format、code、duration_sec
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

        # format 提取
        fmt = 'unknown'
        for key in answers:
            m = re.match(r'tutorial-(\w+)-part1', key)
            if m:
                fmt = m.group(1).lower()
                break

        # 找写作任务
        for content in answers.values():
            if not isinstance(content, dict):
                continue
            if content.get('componentName') != 'writing-task-NL':
                continue

            st = content.get('startTime')
            ed = content.get('endTime')
            dur = (ed - st)/1000.0 if st is not None and ed is not None else None

            ans = content.get('answer', {}) or {}
            code = ans.get('code')

            rows.append({
                'participantId': pid,
                'format':        fmt,
                'code':          code,
                'duration_sec':  dur
            })

    df = pd.DataFrame(rows)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    return df


def calculate_text_similarity(user_text: str, reference_text: str) -> float:
    """
    规范化空白后，用 SequenceMatcher 计算 0.0–1.0 相似度。
    """
    def norm(s):
        return " ".join(str(s).split()).lower()
    return difflib.SequenceMatcher(None, norm(user_text), norm(reference_text)).ratio()


def add_writing_metrics(df: pd.DataFrame, correct_text: str) -> pd.DataFrame:
    """
    在 df 上增加：
      - 'similarity'：用户 code 与 correct_text 的相似度  
      - 'answer_length'：用户 code 的字符数  
    不再添加 correct_answer_length 列。
    """
    df2 = df.copy()
    df2['answer_length'] = df2['code'].fillna("").astype(str).apply(len)
    df2['similarity']    = df2['code'].apply(lambda x: calculate_text_similarity(x, correct_text))
    return df2


def get_correct_answer_length(correct_text: str) -> int:
    """
    返回正确答案的字符数（不含首尾额外空白）。
    """
    return len(str(correct_text))


def summarize_writing_by_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 format 分组，计算：
      - avg_duration_sec
      - avg_similarity
      - avg_answer_length
    """
    summary = (
        df.groupby('format')
          .agg(
            avg_duration_sec = ('duration_sec', 'mean'),
            avg_similarity   = ('similarity',   'mean'),
            avg_answer_length= ('answer_length','mean')
          )
          .reset_index()
    )
    return summary


def plot_writing_summary(summary: pd.DataFrame):
    """
    绘制三张并排柱状图：
      - Avg Duration
      - Avg Similarity
      - Avg Answer Length
    """
    fig, axes = plt.subplots(1, 3, figsize=(14,5), constrained_layout=True)
    ax1, ax2, ax3 = axes

    summary.plot(x='format', y='avg_duration_sec',    kind='bar', ax=ax1, legend=False)
    ax1.set_title("Avg Duration (s)");    ax1.tick_params(axis='x', rotation=45)

    summary.plot(x='format', y='avg_similarity',      kind='bar', ax=ax2, legend=False)
    ax2.set_title("Avg Similarity");      ax2.tick_params(axis='x', rotation=45)

    summary.plot(x='format', y='avg_answer_length',   kind='bar', ax=ax3, legend=False)
    ax3.set_title("Avg Answer Length");   ax3.tick_params(axis='x', rotation=45)

    return fig


def extract_writing_tabular_tasks(all_data, format_list=None):
    """
    提取所有 writing-task-tabular-<format> 条目：
      - participantId
      - format               (如 'hjson')
      - code                 (answer['code'])
      - startTime, endTime
      - duration_sec
      - helpButtonClickedCount

    参数:
      all_data   dict, load_quiz_data 的输出
      format_list list[str] 可选, 只保留这些 format

    返回:
      pandas.DataFrame
    """
    rows = []
    for fn, quiz in all_data.items():
        answers = quiz.get('answers', {})

        # participantId
        pid = fn
        for info in answers.values():
            if isinstance(info, dict):
                ansb = info.get('answer', {})
                if isinstance(ansb, dict) and 'prolificId' in ansb:
                    pid = ansb['prolificId']
                    break

        # 查找 writing-task-tabular-<fmt>
        for content in answers.values():
            if not isinstance(content, dict):
                continue
            comp = content.get('componentName','')
            m = re.match(r'^writing-task-tabular-(\w+)$', comp)
            if not m:
                continue
            fmt = m.group(1).lower()
            if format_list and fmt not in format_list:
                continue

            st = content.get('startTime')
            ed = content.get('endTime')
            dur = (ed - st)/1000.0 if st is not None and ed is not None else None

            ansb = content.get('answer', {}) or {}
            code = ansb.get('code')
            help_cnt = content.get('helpButtonClickedCount', None)

            rows.append({
                'participantId': pid,
                'format':        fmt,
                'code':          code,
                'startTime':     st,
                'endTime':       ed,
                'duration_sec':  dur,
                'helpButtonClickedCount': help_cnt
            })

    df = pd.DataFrame(rows)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    return df


def plot_tabular_metrics_by_format(df_tabular, format_list=None):
    """
    接受 df_tabular (含 'format','duration_sec','helpButtonClickedCount'),
    可选 format_list 过滤格式。
    功能：
      1. 打印按 format 汇总的平均 duration_sec 和平均 helpButtonClickedCount
      2. 返回两个 Figure: 
           - fig_duration 显示 average duration
           - fig_help     显示 average helpButtonClickedCount
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # 过滤
    df = df_tabular.copy()
    if format_list is not None:
        df = df[df['format'].isin(format_list)]
    
    # 汇总
    summary = (
        df.groupby('format')
          .agg(
            avg_duration_sec=('duration_sec', 'mean'),
            avg_help_clicks =('helpButtonClickedCount', 'mean')
          )
          .reset_index()
    )
    print("Summary by format:")
    print(summary)
    
    # 绘制 average duration
    fig_duration, ax1 = plt.subplots(figsize=(8,5), constrained_layout=True)
    ax1.bar(summary['format'], summary['avg_duration_sec'])
    ax1.set_title("Average Duration by Format")
    ax1.set_xlabel("Format")
    ax1.set_ylabel("Duration (s)")
    ax1.tick_params(axis='x', rotation=45)
    
    # 绘制 average help clicks
    fig_help, ax2 = plt.subplots(figsize=(8,5), constrained_layout=True)
    ax2.bar(summary['format'], summary['avg_help_clicks'])
    ax2.set_title("Average HelpButtonClickedCount by Format")
    ax2.set_xlabel("Format")
    ax2.set_ylabel("Help Clicks")
    ax2.tick_params(axis='x', rotation=45)
    
    return fig_duration, fig_help



def extract_modifying_tabular_tasks(all_data, format_list=None):
    """
    提取所有 modifying-task-tabular-<format>-<n> 条目：
      - participantId
      - format               (如 'hjson')
      - code                 (answer['code'])
      - startTime, endTime
      - duration_sec
      - helpButtonClickedCount

    参数:
      all_data    dict, load_quiz_data 的输出
      format_list list[str] 可选, 只保留这些 format

    返回:
      pandas.DataFrame
    """
    rows = []
    pattern = re.compile(r'^modifying-task-tabular-([^-]+)-\d+$')

    for fn, quiz in all_data.items():
        answers = quiz.get('answers', {})

        # 获取 participantId
        pid = fn
        for info in answers.values():
            if isinstance(info, dict):
                ansb = info.get('answer', {})
                if isinstance(ansb, dict) and 'prolificId' in ansb:
                    pid = ansb['prolificId']
                    break

        # 查找符合 pattern 的 task
        for content in answers.values():
            if not isinstance(content, dict):
                continue
            comp = content.get('componentName', '')
            m = pattern.match(comp)
            if not m:
                continue
            fmt = m.group(1).lower()
            if format_list and fmt not in format_list:
                continue

            st = content.get('startTime')
            ed = content.get('endTime')
            dur = (ed - st) / 1000.0 if st is not None and ed is not None else None

            ansb = content.get('answer', {}) or {}
            code = ansb.get('code')
            help_cnt = content.get('helpButtonClickedCount', None)

            rows.append({
                'participantId': pid,
                'format':        fmt,
                'task':         comp,
                'code':          code,
                'startTime':     st,
                'endTime':       ed,
                'duration_sec':  dur,
                'helpButtonClickedCount': help_cnt
            })

    df = pd.DataFrame(rows)
    df['duration_sec'] = pd.to_numeric(df['duration_sec'], errors='coerce')
    return df