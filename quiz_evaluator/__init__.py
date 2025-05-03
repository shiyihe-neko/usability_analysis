# Creating a new package `quiz_evaluator`
# Save this file as `quiz_evaluator/__init__.py`

import pandas as pd
import re
from collections import Counter

def evaluate_quiz_answers_from_tutorial(all_data):
    """
    Iterate through all_data to evaluate tutorial quiz answers:
      - Extract participantId from prolificId or filename
      - For each tutorial part (part1/part2), compare user answers against correct answers
      - Count wrong attempts and distributions
      - Return a DataFrame of quiz results per participant and quiz key
    """
    quiz_results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # Extract participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                answer_block = task_info.get('answer', {})
                if isinstance(answer_block, dict) and 'prolificId' in answer_block:
                    participant_id = answer_block['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # Process each tutorial quiz task
        for task_key, task_info in answers.items():
            if not isinstance(task_info, dict):
                continue
            if not re.match(r"tutorial-\w+-part[12]", task_key):
                continue

            correct_list = task_info.get("correctAnswer", [])
            if not correct_list or not isinstance(correct_list[0], dict):
                continue
            quiz_id = correct_list[0].get("id")
            correct_answer = correct_list[0].get("answer", [])
            correct_set = set(correct_answer)

            answer_block = task_info.get("answer", {})
            user_final = answer_block.get(quiz_id, [])
            is_correct = set(user_final) == correct_set

            attempts = task_info.get("incorrectAnswers", {}).get(quiz_id, [])
            counter = Counter()
            for at in attempts:
                counter.update(at)

            wrong_dist = {c: cnt for c, cnt in counter.items() if c not in correct_set}
            wrong_count = sum(wrong_dist.values())

            quiz_results.append({
                "participantId": participant_id,
                "quiz_key": task_key,
                "correct_answer": correct_answer,
                "user_final_answer": user_final,
                "is_correct": is_correct,
                "num_wrong_attempts": len(attempts),
                "all_wrong_attempts_list": attempts,
                "all_wrong_attempts_frequency": dict(counter),
                "wrong_choice_distribution": wrong_dist,
                "wrong_choice_count": wrong_count
            })

    return pd.DataFrame(quiz_results)


def analyze_nasa_and_post_surveys(all_data):
    """
    Analyze NASA-TLX and post-task surveys:
      - Extract participantId and format from tutorial keys
      - Build two DataFrames: NASA-TLX dimensions and post-task survey responses
    Returns:
      df_nasa: DataFrame of NASA-TLX results per participant
      df_post_survey: DataFrame of post-task survey responses per participant
    """
    nasa_rows = []
    post_survey_rows = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # Extract participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                ans = task_info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    participant_id = ans['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # Extract format
        format_name = None
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break
        format_name = format_name or "unknown"

        # NASA-TLX
        nasa_key = '$nasa-tlx.co.nasa-tlx'
        if nasa_key in answers:
            info = answers[nasa_key]
            ans = info.get('answer', {})
            start, end = info.get('startTime'), info.get('endTime')
            duration = (end - start)/1000.0 if start and end else None
            row = {
                'participantId': participant_id,
                'format': format_name,
                'startTime': start,
                'endTime': end,
                'duration_sec': duration
            }
            for dim in ['mental-demand','physical-demand','temporal-demand',
                        'performance','effort','frustration']:
                row[dim] = ans.get(dim)
            nasa_rows.append(row)

        # Post-task survey
        post_key = 'post-task-survey-tlx'
        if post_key in answers:
            info = answers[post_key]
            ans = info.get('answer', {})
            start, end = info.get('startTime'), info.get('endTime')
            duration = (end - start)/1000.0 if start and end else None
            row = {
                'participantId': participant_id,
                'format': format_name,
                'startTime': start,
                'endTime': end,
                'duration_sec': duration
            }
            for q in ['q7','q8','q9','q9-other','q10','q11',
                      'q13','q13-other','q14','q14-other']:
                row[q] = ans.get(q)
            # Expand q12 block
            q12 = ans.get('q12', {})
            if isinstance(q12, dict):
                for fmt, lvl in q12.items():
                    row[f'q12_{fmt}'] = lvl
            post_survey_rows.append(row)

    df_nasa = pd.DataFrame(nasa_rows)
    df_post = pd.DataFrame(post_survey_rows)
    return df_nasa, df_post


def extract_post_task_questions(all_data):
    """
    Extract post-task questionnaire responses:
      - For keys ending with '_post-task-question', capture difficulty and confidence
      - Include start/end times and calculate duration
    Returns a DataFrame of post-task question results per participant and task
    """
    rows = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})

        # Extract participantId
        participant_id = None
        for task_info in answers.values():
            if isinstance(task_info, dict):
                ans = task_info.get('answer', {})
                if isinstance(ans, dict) and 'prolificId' in ans:
                    participant_id = ans['prolificId']
                    break
        if participant_id is None:
            participant_id = file_name

        # Extract format
        format_name = None
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break
        format_name = format_name or "unknown"

        # Collect post-task question entries
        for key, content in answers.items():
            if "_post-task-question" not in key or not isinstance(content, dict):
                continue
            base_task = key.replace("_post-task-question", "")
            ans = content.get('answer', {})
            difficulty = ans.get('difficulty')
            confidence = ans.get('confidence')
            start, end = content.get('startTime'), content.get('endTime')
            duration = (end - start)/1000.0 if start and end else None

            rows.append({
                "participantId": participant_id,
                "format": format_name,
                "task": base_task,
                "startTime": start,
                "endTime": end,
                "duration_sec": duration,
                "difficulty": difficulty,
                "confidence": confidence
            })

    return pd.DataFrame(rows)

