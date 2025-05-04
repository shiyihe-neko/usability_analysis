# duration_processor/__init__.py

import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

def load_quiz_data(folder_path):
    """
    Iterate over all .json files in folder_path:
      - Include only those with quiz['completed'] == True
      - Rename answer keys that have numeric suffixes (_1, _2, …)
      - Apply renaming logic for post-task-question and post-task-survey keys
    Returns a dict mapping each filename (without extension) to its processed quiz data dict.
    """
    def extract_suffix(key):
        m = re.search(r'_(\d+)$', key)
        return int(m.group(1)) if m else 0

    def remove_suffix(key):
        return re.sub(r'_(\d+)$', '', key)

    all_data = {}
    for fn in os.listdir(folder_path):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(folder_path, fn)
        try:
            with open(path, encoding='utf-8') as f:
                quiz = json.load(f)
        except json.JSONDecodeError:
            continue

        if not quiz.get('completed', False):
            continue

        key_name = os.path.splitext(fn)[0]
        all_data[key_name] = quiz

        answers = quiz.get('answers', {})
        if not isinstance(answers, dict):
            continue

        # rename
        sorted_keys = sorted(answers.keys(), key=extract_suffix)
        new_answers = {}
        last_task = None
        for i, old in enumerate(sorted_keys):
            base = remove_suffix(old)

            if base == 'post-task-question':
                if last_task:
                    new_key = f"{last_task}_post-task-question"
                else:
                    new_key = base
            elif base.startswith('post-task-survey'):
                if i>0:
                    prev = sorted_keys[i-1]
                    prev_base = remove_suffix(prev)
                    suffix = prev_base[prev_base.rfind('-'):] if '-' in prev_base else ''
                    new_key = base + suffix
                else:
                    new_key = base
                last_task = None
            else:
                new_key = base
                last_task = base

            new_answers[new_key] = answers[old]

        quiz['answers'] = new_answers

    return all_data


def extract_format_time(all_data):
    """
    Parameters:
      all_data (dict): The dictionary returned by load_quiz_data.

    Returns:
      df_task_level (pandas.DataFrame):
        A DataFrame with one row per task, including columns for start time, end time, duration, and format.
      df_participant_level (pandas.DataFrame):
        A DataFrame with one row per participant, including columns for total duration and format.
    """
    task_rows = []
    part_rows = []

    for fn, quiz in all_data.items():
        answers = quiz.get('answers', {})
        # look for participantId
        pid = fn
        for info in answers.values():
            if isinstance(info, dict):
                ans = info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    pid = ans['prolificId']
                    break

        # extract  format
        current_format = None
        total_sec = 0
        temp = []
        for name, info in answers.items():
            if not isinstance(info, dict):
                continue
            st = info.get('startTime')
            ed = info.get('endTime')
            if st is not None and ed is not None:
                dur = (ed - st)/1000.0
            else:
                dur = None

            # tutorial-<fmt>-part1
            if current_format is None:
                m = re.match(r'tutorial-(\w+)-part1', name)
                if m:
                    current_format = m.group(1).lower()

            temp.append({
                'participantId': pid,
                'task': name,
                'startTime': st,
                'endTime': ed,
                'duration_sec': dur,
                'duration_min': dur/60 if dur is not None else None
            })
            if dur:
                total_sec += dur

        # add format 
        fmt = current_format or 'unknown'
        for row in temp:
            row['format'] = fmt
            task_rows.append(row)

        part_rows.append({
            'participantId': pid,
            'format': fmt,
            'total_duration_sec': round(total_sec,3),
            'total_duration_min': round(total_sec/60,2)
        })

    df_task = pd.DataFrame(task_rows)
    df_part = pd.DataFrame(part_rows)
    return df_task, df_part


def sanitize_task_names(df):
    """
    Remove any '-<format>' segment from the 'task' column values.
    For example, 'modifying-task-tabular-json5-4' → 'modifying-task-4'.
    """
    df = df.copy()
    def clean(name, fmt):
        if fmt and fmt!='unknown':
            return name.replace(f"-{fmt}", "")
        return name

    df['task'] = df.apply(lambda r: clean(r['task'], r['format']), axis=1)
    return df


def summarize_participant_by_format(df_participant, metric='total_duration_min'):
    """
    Group the DataFrame by 'format', calculate the mean of the specified metric,
    rename the resulting column to 'average_<metric>', and sort the DataFrame
    in descending order by that new column.
    """
    # 1. calculate mean
    summary = df_participant.groupby('format')[metric].mean().reset_index()
    # 2. rename column - average_<metric>
    avg_col = f'average_{metric}'
    summary = summary.rename(columns={metric: avg_col})
    # 3. sourt by new column
    return summary.sort_values(by=avg_col, ascending=False)

def plot_participant_time_by_format(df_summary,
                                    metric='average_total_duration_min',
                                    title='Average Participant Time by Format',
                                    xlabel='Format',
                                    ylabel=None):
    """
    Plot a bar chart of average participant times by format,
    supporting custom chart title and axis labels.
    """
    fig, ax = plt.subplots()
    ax.bar(df_summary['format'], df_summary[metric])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f'Average {metric}')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def filter_tasks_by_list(df_task, task_list):
    """
    Filter the df_task DataFrame to include only rows for tasks in task_list.
    Returns the filtered DataFrame for inspection.
    """
    return df_task[df_task['task'].isin(task_list)].copy()


def summarize_tasks_by_format(df_task, task_list, metric='duration_min', mode='overall'):
    """
    Filter the df_task DataFrame to include only tasks in task_list, then summarize based on mode:
      - 'overall': group by 'format' and calculate the mean of the specified metric across all selected tasks,
                   renaming the result column to 'average_<metric>'.
      - 'by_task': group by both 'format' and 'task' and calculate the mean of the specified metric for each combination,
                   renaming the result column to 'average_<metric>'.
    Return the summary DataFrame sorted according to the chosen mode.
    """
    df_filtered = df_task[df_task['task'].isin(task_list)].copy()
    avg_col = f'average_{metric}'
    
    if mode == 'overall':
        summary = (
            df_filtered
            .groupby('format')[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: avg_col})
        )
        return summary.sort_values(by=avg_col, ascending=False)
    
    elif mode == 'by_task':
        summary = (
            df_filtered
            .groupby(['format', 'task'])[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: avg_col})
        )
        # sort by format and task
        return summary.sort_values(['format', 'task'])
    
    else:
        raise ValueError("mode must be 'overall' or 'by_task'")



def plot_tasks_time_by_format(df_summary,
                              metric='average_duration_min',
                              mode='overall',
                              title='Average Time by Format',
                              xlabel='Format',
                              ylabel=None):
    """
    Plot a bar chart based on the output of summarize_tasks_by_format:
      - overall: a single bar for each format
      - by_task: grouped bar chart with one bar per task under each format
    Supports custom chart title and axis labels.
    """
    fig, ax = plt.subplots()
    if mode == 'overall':
        ax.bar(df_summary['format'], df_summary[metric])
    else:  # by_task
        # pivot：row=format, column=task, value=metric
        pivot = df_summary.pivot(index='format', columns='task', values=metric)
        pivot.plot(kind='bar', ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f'Average {metric}')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def participant_format_statistics(df_participant):
    format_stat = df_participant['format'].value_counts()
    total_participant = len(df_participant)
    print(f"Total number of valid participants: {total_participant}")
    return format_stat
