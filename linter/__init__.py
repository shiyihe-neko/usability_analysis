# writing_tabular_analysis.py

import re
import json
import json5            # pip install json5
import hjson            # pip install hjson
import yaml             # pip install pyyaml
import toml             # pip install toml
import xml.etree.ElementTree as ET
import defusedxml.ElementTree as DET  # pip install defusedxml
import xmltodict        # pip install xmltodict
import tempfile
import subprocess
import pandas as pd
import numpy as np
from tree_sitter import Language, Parser  # pip install tree_sitter
import zss              # pip install zss
from writing_evaluator import extract_writing_tabular_tasks
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# 1) 语法校验
# -----------------------------------------------------------------------------

def validate_tabular_syntax_all(
    df_tabular: pd.DataFrame,
    code_col: str = 'code',
    format_col: str = 'format',
    strict_col: str = 'syntax_strict',
    loose_col: str = 'syntax_loose',
    repair_col: str = 'syntax_repair'
) -> pd.DataFrame:
    """
    针对 tabular writing task 的 code 列做三步语法校验：
      1) 严格解析 → strict_col (1/0)
      2) 对严格失败的，做宽松解析 → loose_col (1/0)，严格成功行留 NA
      3) 对既不严格也不宽松的，再尝试自动修复 → repair_col (1/0)，否则 NA

    JSONC 在“严格”阶段会自动：
      - 去掉 /*…*/ 和 //… 注释
      - 去除尾逗号

    返回原表的拷贝，并附加三列校验结果，不修改原 DataFrame。
    """
    # ---- JSONC 预处理（剥注释 + 去尾逗号） ---- #
    def preprocess_jsonc(text: str) -> str:
        no_block = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        no_line  = re.sub(r'//.*?$',       '', no_block, flags=re.MULTILINE)
        clean    = re.sub(r',\s*(?=[}\]])', '', no_line)
        return clean

    # ---- 严格解析器 ---- #
    STRICT = {
        'json':  lambda s: json.loads(s),
        'jsonc': lambda s: json5.loads(s),
        'json5': lambda s: json5.loads(s),
        'hjson': lambda s: hjson.loads(s, strict=True),
        'yaml':  lambda s: yaml.safe_load(s),
        'toml':  lambda s: toml.loads(s),
        'xml':   lambda s: ET.fromstring(s)
    }

    # ---- 宽松解析器 ---- #
    LOOSE = {
        'json':  lambda s: json5.loads(s),
        'jsonc': lambda s: json5.loads(s),
        'json5': lambda s: hjson.loads(s),
        'hjson': lambda s: hjson.loads(s),
        'yaml':  lambda s: yaml.load(s, Loader=yaml.FullLoader),
        'toml':  lambda s: toml.loads(s),
        'xml':   lambda s: DET.fromstring(s)
    }

    df = df_tabular.copy()

    # 1) 严格解析
    def _check_strict(row):
        fmt = str(row[format_col]).lower()
        code = row.get(code_col, "") or ""
        parser = STRICT.get(fmt)
        try:
            if not parser: 
                return 0
            parser(code)
            return 1
        except:
            return 0

    df[strict_col] = df.apply(_check_strict, axis=1)

    # 2) 宽松解析（仅对 strict==0）
    df[loose_col] = pd.NA
    def _check_loose(row):
        fmt = str(row[format_col]).lower()
        code = row.get(code_col, "") or ""
        parser = LOOSE.get(fmt)
        try:
            if not parser:
                return 0
            parser(code)
            return 1
        except:
            return 0

    mask_loose = df[strict_col] == 0
    df.loc[mask_loose, loose_col] = (
        df.loc[mask_loose].apply(_check_loose, axis=1).astype("Int64")
    )

    # 3) 自动修复（strict==0 且 loose==0）
    df[repair_col] = pd.NA
    def _attempt_repair(row):
        fmt = str(row[format_col]).lower()
        code = row.get(code_col, "") or ""
        # 如果已经 strict 或 loose 通过，就不用修复
        if row[strict_col] == 1 or row.get(loose_col) == 1:
            return pd.NA
        try:
            # 宽松解析得到结构
            data = LOOSE[fmt](code)
            # 按格式 dump 回标准化字符串
            if fmt in ('json','jsonc','json5','hjson'):
                rep = json.dumps(data)
            elif fmt == 'yaml':
                rep = yaml.safe_dump(data)
            elif fmt == 'toml':
                rep = toml.dumps(data)
            elif fmt == 'xml':
                rep = ET.tostring(data, encoding='unicode')
            else:
                return 0
            # 再次严格解析
            STRICT[fmt](rep)
            return 1
        except:
            return 0

    mask_repair = (df[strict_col] == 0) & (df[loose_col] == 0)
    df.loc[mask_repair, repair_col] = (
        df.loc[mask_repair].apply(_attempt_repair, axis=1).astype("Int64")
    )

    return df


def summarize_syntax_pass_rates(df: pd.DataFrame,
                                format_col: str = 'format',
                                strict_col: str = 'syntax_strict',
                                loose_col: str = 'syntax_loose',
                                repair_col: str = 'syntax_repair') -> pd.DataFrame:
    """
    对语法校验结果 DataFrame 进行汇总：
    - 每种 format 的 strict / loose / repair 三种通过率（平均值）
    - loose_pass_rate 仅计算 strict==0 的样本
    - repair_pass_rate 仅计算 strict==0 且 loose==0 的样本
    """
    df = df.copy()

    # strict_pass_rate：直接 groupby mean
    strict_rates = df.groupby(format_col)[strict_col].mean().rename('strict_pass_rate')

    # loose_pass_rate：仅在 strict==0 时统计
    mask_loose = df[strict_col] == 0
    loose_rates = (
        df[mask_loose]
        .groupby(format_col)[loose_col]
        .mean()
        .rename('loose_pass_rate')
    )

    # repair_pass_rate：仅在 strict==0 且 loose==0 时统计
    mask_repair = (df[strict_col] == 0) & (df[loose_col] == 0)
    repair_rates = (
        df[mask_repair]
        .groupby(format_col)[repair_col]
        .mean()
        .rename('repair_pass_rate')
    )

    # 合并三个指标
    summary = (
        pd.concat([strict_rates, loose_rates, repair_rates], axis=1)
          .fillna(0)
          .reset_index()
    )

    return summary


def plot_syntax_pass_rates(summary_df: pd.DataFrame):
    """
    输入 summary_df，其中包含 'format', 'strict_pass_rate', 'loose_pass_rate', 'repair_pass_rate' 三列，
    画出每种 format 的三种通过率的对比柱状图。
    """
    df_plot = summary_df.set_index('format')[['strict_pass_rate', 'loose_pass_rate', 'repair_pass_rate']]

    df_plot.plot(kind='bar', figsize=(12, 6))
    plt.title("Syntax Pass Rates by Format")
    plt.ylabel("Pass Rate")
    plt.xlabel("Format")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.legend(title="Pass Type")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 3) Git-diff 比对写作任务改动

def preprocess_jsonc(text: str) -> str:
    no_block = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    no_line = re.sub(r'//.*?$', '', no_block, flags=re.MULTILINE)
    clean = re.sub(r',\s*(?=[}\]])', '', no_line)
    return clean

STRICT_PARSERS = {
    'json':  lambda s: json.loads(s),
    'jsonc': lambda s: json.loads(preprocess_jsonc(s)),
    'json5': lambda s: json5.loads(s),
    'hjson': lambda s: hjson.loads(s, strict=True),
    'yaml':  lambda s: yaml.safe_load(s),
    'toml':  lambda s: toml.loads(s),
    'xml':   lambda s: ET.fromstring(s)
}

def git_diff_string(str1: str, str2: str) -> dict:
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f2:
        f1.write(str1)
        f2.write(str2)
        f1_path, f2_path = f1.name, f2.name

    result = subprocess.run(['git', 'diff', '--no-index', f1_path, f2_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    diff_output = result.stdout
    added_lines = sum(1 for line in diff_output.splitlines() if line.startswith('+') and not line.startswith('+++'))
    removed_lines = sum(1 for line in diff_output.splitlines() if line.startswith('-') and not line.startswith('---'))

    return {
        "diff_text": diff_output if diff_output.strip() else "No differences found.",
        "num_added_lines": added_lines,
        "num_removed_lines": removed_lines,
        "total_changes": added_lines + removed_lines
    }

# —— 正确答案序列化 —— #
def serialize_correct_tabular_data(correct_data: dict, fmt: str) -> str:
    f = fmt.lower()
    try:
        if f in ['json', 'jsonc', 'json5', 'hjson']:
            return json.dumps(correct_data, indent=2, sort_keys=True)
        elif f == 'toml':
            return toml.dumps(correct_data)
        elif f == 'yaml':
            return yaml.dump(correct_data, sort_keys=False)
        elif f == 'xml':
            # 简化处理：目前还是用 json 输出
            return json.dumps(correct_data, indent=2, sort_keys=True)
        else:
            return json.dumps(correct_data, indent=2, sort_keys=True)
    except Exception as e:
        return f"ERROR: {e}"

# —— 提取 tabular 任务 —— #
def extract_writing_tabular_tasks(all_data: dict) -> pd.DataFrame:
    rows = []
    for fn, quiz in all_data.items():
        answers = quiz.get("answers", {})
        pid = fn
        for v in answers.values():
            if isinstance(v, dict) and isinstance(v.get('answer'), dict):
                prof = v['answer'].get('prolificId')
                if prof:
                    pid = prof
                    break
        fmt = None
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                fmt = m.group(1).lower()
                break
        for key, content in answers.items():
            if key.startswith("writing-task-tabular-") and isinstance(content, dict):
                code = content.get('answer', {}).get('code')
                if code:
                    task_fmt = key.split("writing-task-tabular-")[1].split("-")[0].lower()
                    rows.append({'participantId': pid, 'format': task_fmt or fmt or 'unknown',
                                 'task': key, 'code': code})
    return pd.DataFrame(rows)

# —— 主函数 —— #
def analyze_tabular_writing_diff(all_data: dict,
                                 correct_data: dict) -> pd.DataFrame:
    df_tab = extract_writing_tabular_tasks(all_data)

    results = []
    for _, row in df_tab.iterrows():
        fmt = row['format']
        user_code = (row['code'] or "").strip()
        correct_code = serialize_correct_tabular_data(correct_data, fmt)

        diff = git_diff_string(correct_code, user_code)

        # 尝试语法检查（不用于筛选，仅供分析）
        parser = STRICT_PARSERS.get(fmt)
        try:
            valid = 1 if parser and parser(user_code) else 0
        except:
            valid = 0

        results.append({
            "participantId":     row['participantId'],
            "format":            fmt,
            "task":              row['task'],
            "syntax_valid":      valid,
            "num_added_lines":   diff['num_added_lines'],
            "num_removed_lines": diff['num_removed_lines'],
            "total_changes":     diff['total_changes'],
            "diff_text":         diff['diff_text']
        })

    return pd.DataFrame(results)

def plot_avg_total_changes_by_format(df_diff: pd.DataFrame):
    """
    对 diff 分析结果中的 total_changes 按 format 分组平均，并绘制柱状图。
    """
    avg_changes = (
        df_diff.groupby('format')['total_changes']
               .mean()
               .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    avg_changes.plot(kind='bar')
    plt.title("Average Total Changes per Format")
    plt.ylabel("Average Total Changes")
    plt.xlabel("Format")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 4) Tree-sitter AST tree-edit distance 比对
# -----------------------------------------------------------------------------

# # 初始化 Tree-sitter JSON parser
# JSON_LANGUAGE = Language('build/my-languages.so', 'json')  # 确保此路径有效
# TS_PARSER = Parser()
# TS_PARSER.set_language(JSON_LANGUAGE)

# # NodeWrapper 用于 zss tree-edit distance 比较
# class NodeWrapper:
#     def __init__(self, node, src_bytes):
#         self.node = node
#         self.src = src_bytes
#     def get_label(self):
#         txt = self.src[self.node.start_byte:self.node.end_byte].decode('utf-8').strip()
#         return f"{self.node.type}:{txt}"
#     def get_children(self):
#         return [NodeWrapper(c, self.src) for c in self.node.children]

# def tree_size(node: NodeWrapper) -> int:
#     return 1 + sum(tree_size(c) for c in node.get_children())

# # JSONC 注释移除
# def remove_jsonc_comments(text: str) -> str:
#     no_block = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
#     no_line  = re.sub(r'//.*?$', '', no_block, flags=re.MULTILINE)
#     return no_line

# # 将用户代码转换成 JSON 字符串（自动修复版本）
# def convert_to_json_string_fixed(code_str: str, fmt: str) -> str:
#     f = fmt.lower()
#     try:
#         if f in ["json", "jsonc"]:
#             code_str = remove_jsonc_comments(code_str)
#             code_str = re.sub(r",\s*([\]}])", r"\1", code_str)
#             obj = json.loads(code_str)
#         elif f == "json5":
#             code_str = re.sub(r",\s*([\]}])", r"\1", code_str)
#             obj = json5.loads(code_str)
#         elif f == "hjson":
#             obj = hjson.loads(code_str)
#         elif f == "yaml":
#             try:
#                 obj = yaml.safe_load(code_str)
#             except:
#                 obj = yaml.load(code_str, Loader=yaml.FullLoader)
#         elif f == "toml":
#             code_str = re.sub(r"\n\s*\n", "\n", code_str)
#             if "'" in code_str or '"' in code_str:
#                 code_str = re.sub(r"(['\"])(.*?)\1?", r'"\2"', code_str)
#             obj = toml.loads(code_str)
#         elif f == "xml":
#             txt = code_str.strip()
#             if not txt.startswith("<root>"):
#                 txt = f"<root>{txt}</root>"
#             try:
#                 obj = xmltodict.parse(txt)
#             except:
#                 txt_fixed = re.sub(r"<([^/>]+)>([^<>]*)<\1>", r"<\1>\2</\1>", txt)
#                 obj = xmltodict.parse(txt_fixed)
#         else:
#             raise ValueError(f"Unsupported format: {fmt}")
#     except Exception as e:
#         raise ValueError(f"[Auto-fix failed] Parse error for format={fmt}: {e}")
#     return json.dumps(obj, indent=2, sort_keys=True)

# # 将标准答案转换成 JSON 字符串（格式一致）
# def prepare_gold_structure(correct_data: dict, fmt: str) -> str:
#     if fmt.lower() == 'xml':
#         return json.dumps({"root": correct_data}, indent=2, sort_keys=True)
#     return json.dumps(correct_data, indent=2, sort_keys=True)

# # 计算 Tree-edit distance 相似度
# def compare_structures_tree_edit_distance(user_code: str, gold_json: str, fmt: str) -> dict:
#     try:
#         user_json = convert_to_json_string_fixed(user_code, fmt)
#     except Exception as e:
#         return {"ted": None, "normalized_ted": None, "similarity_score": None, "error": str(e)}
    
#     src1, src2 = bytes(gold_json, 'utf-8'), bytes(user_json, 'utf-8')
#     t1, t2 = TS_PARSER.parse(src1), TS_PARSER.parse(src2)
#     w1, w2 = NodeWrapper(t1.root_node, src1), NodeWrapper(t2.root_node, src2)

#     ted = zss.simple_distance(w1, w2,
#                               get_children=lambda n: n.get_children(),
#                               get_label=lambda n: n.get_label())
#     size1, size2 = tree_size(w1), tree_size(w2)
#     max_size = max(size1, size2) or 1
#     n_ted = ted / max_size
#     return {
#         "ted": ted,
#         "normalized_ted": round(n_ted, 4),
#         "similarity_score": round(1 - n_ted, 4),
#         "error": None
#     }

# # 主函数：批量处理所有参与者的结构比对任务
# def batch_tree_distance_analysis(all_data: dict, correct_tabular_data: dict) -> pd.DataFrame:
#     rows = []
#     for fn, quiz in all_data.items():
#         answers = quiz.get('answers', {})
#         pid = fn
#         for v in answers.values():
#             if isinstance(v, dict):
#                 ans = v.get('answer', {})
#                 if isinstance(ans, dict) and 'prolificId' in ans:
#                     pid = ans['prolificId']
#                     break

#         fmt = None
#         for k in answers:
#             m = re.match(r"tutorial-(\w+)-part1", k)
#             if m:
#                 fmt = m.group(1).lower()
#                 break
#         if not fmt:
#             for k in answers:
#                 if k.startswith("writing-task-tabular-"):
#                     fmt = k.split("writing-task-tabular-")[1].split("-")[0].lower()
#                     break
#         if not fmt:
#             continue  # 无法确定格式

#         gold_json = prepare_gold_structure(correct_tabular_data, fmt)

#         for key, content in answers.items():
#             if not key.startswith("writing-task-tabular-"):
#                 continue
#             code = content.get('answer', {}).get('code')
#             if not code:
#                 continue
#             res = compare_structures_tree_edit_distance(code, gold_json, fmt)
#             rows.append({
#                 "participantId": pid,
#                 "format": fmt,
#                 "task": key,
#                 **res
#             })
#     return pd.DataFrame(rows)

# def plot_similarity_score_by_format(df_ted: pd.DataFrame):
#     """
#     平均 similarity_score 按 format 分组画柱状图
#     """
#     avg_sim = (
#         df_ted.groupby("format")["similarity_score"]
#               .mean()
#               .sort_values(ascending=False)
#     )

#     plt.figure(figsize=(10, 5))
#     avg_sim.plot(kind='bar')
#     plt.title("Average Structure Similarity Score (Tree-edit Distance)")
#     plt.ylabel("Similarity Score")
#     plt.xlabel("Format")
#     plt.xticks(rotation=45)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()


#     "json": Language('build/my-languages.so', 'json'),
#     "yaml": Language('build/my-languages.so', 'yaml'),
#     "toml": Language('build/my-languages.so', 'toml'),
#     "xml": Language('build/my-languages.so', 'xml'),
#     # 其它格式（json5/jsonc/hjson）建议先预处理成json后使用json parser
# }

# def get_parser_for_format(fmt: str) -> Parser:
#     if fmt not in LANGUAGES:
#         raise ValueError(f"Unsupported format: {fmt}")
#     parser = Parser()
#     parser.set_language(LANGUAGES[fmt])
#     return parser


# class NodeWrapper:
#     def __init__(self, node, src_bytes):
#         self.node = node
#         self.src = src_bytes
#     def get_label(self):
#         text = self.src[self.node.start_byte:self.node.end_byte].decode('utf-8').strip()
#         return f"{self.node.type}:{text}"
#     def get_children(self):
#         return [NodeWrapper(c, self.src) for c in self.node.children]

# def tree_size(node):
#     return 1 + sum(tree_size(c) for c in node.get_children())

# def compare_ast_by_tree_sitter(user_code: str, gold_code: str) -> dict:
#     try:
#         # 强制转为 JSON 字符串，避免因格式不一致失败
#         gold_json = json.dumps(json.loads(gold_code), indent=2, sort_keys=True)
#         user_json = json.dumps(json.loads(user_code), indent=2, sort_keys=True)
#         src1, src2 = gold_json.encode(), user_json.encode()
#         tree1, tree2 = TS_PARSER.parse(src1), TS_PARSER.parse(src2)
#         node1, node2 = NodeWrapper(tree1.root_node, src1), NodeWrapper(tree2.root_node, src2)
#         ted = zss.simple_distance(node1, node2,
#                                   get_children=lambda n: n.get_children(),
#                                   get_label=lambda n: n.get_label())
#         max_size = max(tree_size(node1), tree_size(node2)) or 1
#         return {
#             "ted": ted,
#             "normalized_ted": round(ted / max_size, 4),
#             "similarity_score": round(1 - ted / max_size, 4),
#             "error": None
#         }
#     except Exception as e:
#         return {"ted": None, "normalized_ted": None, "similarity_score": None, "error": f"[TreeSitter] {e}"}

# def batch_tree_distance_analysis_tree_only(all_data: dict, correct_data: dict) -> pd.DataFrame:
#     """
#     使用 Tree-sitter AST 比较结构差异，返回每个参与者与标准答案的结构相似度。
#     """
#     results = []

#     gold_code = json.dumps(correct_data, indent=2, sort_keys=True)  # 转为 JSON 字符串，统一格式

#     for file_name, quiz_data in all_data.items():
#         answers = quiz_data.get('answers', {})

#         # 提取 participantId
#         pid = file_name
#         for task_info in answers.values():
#             if isinstance(task_info, dict):
#                 ans = task_info.get('answer', {})
#                 if isinstance(ans, dict) and 'prolificId' in ans:
#                     pid = ans['prolificId']
#                     break

#         # 提取 format（从 tutorial-xxx-part1 推断）
#         fmt = "unknown"
#         for k in answers:
#             m = re.match(r"tutorial-(\w+)-part1", k)
#             if m:
#                 fmt = m.group(1).lower()
#                 break

#         # 遍历 writing-task-tabular-xxx
#         for key, content in answers.items():
#             if not key.startswith("writing-task-tabular-"):
#                 continue
#             answer_block = content.get('answer', {})
#             code = answer_block.get('code') if isinstance(answer_block, dict) else None
#             if not code:
#                 continue

#             res = compare_ast_by_tree_sitter(code, gold_code)

#             results.append({
#                 "participantId": pid,
#                 "format": fmt,
#                 "task": key,
#                 **res
#             })

#     return pd.DataFrame(results)

# 设置 Tree-sitter JSON parser（请确保 build/my-languages.so 已构建）
# try:
#     JSON_LANGUAGE = Language('build/my-languages.so', 'json')
#     parser = Parser()
#     parser.set_language(JSON_LANGUAGE)
# except Exception as e:
#     print(f"初始化Tree-sitter解析器失败: {e}")
#     print("请确保已编译tree-sitter语言库到build/my-languages.so")

# # Tree node wrapper for ZSS
# class NodeWrapper:
#     def __init__(self, node, source):
#         self.node = node
#         self.source = source

#     def get_label(self):
#         try:
#             return f"{self.node.type}:{self.source[self.node.start_byte:self.node.end_byte].decode('utf-8').strip()}"
#         except Exception:
#             return f"{self.node.type}:error"

#     def get_children(self):
#         return [NodeWrapper(child, self.source) for child in self.node.children]

# def tree_size(node):
#     """计算树的节点数量"""
#     return 1 + sum(tree_size(child) for child in node.get_children())

# def clean_comments(code_str):
#     """清除代码中的注释"""
#     # 清除单行注释
#     code_str = re.sub(r"//.*?$", "", code_str, flags=re.MULTILINE)
#     # 清除多行注释
#     code_str = re.sub(r"/\*.*?\*/", "", code_str, flags=re.DOTALL)
#     return code_str

# def convert_to_json_string(code_str, fmt):
#     """转换任意支持格式为标准 JSON 字符串（供 tree-sitter 使用）"""
#     if not code_str or not isinstance(code_str, str):
#         raise ValueError(f"Invalid code string: {type(code_str)}")
    
#     code_str = code_str.strip()
#     fmt = fmt.lower()
    
#     try:
#         if fmt == "json":
#             parsed = json.loads(code_str)
#         elif fmt == "jsonc":
#             cleaned_code = clean_comments(code_str)
#             parsed = json.loads(cleaned_code)
#         elif fmt == "json5":
#             parsed = json5.loads(code_str)
#         elif fmt == "hjson":
#             parsed = hjson.loads(code_str)
#         elif fmt == "yaml" or fmt == "yml":
#             parsed = yaml.safe_load(code_str)
#         elif fmt == "toml":
#             parsed = toml.loads(code_str)
#         elif fmt == "xml":
#             # 确保有根元素
#             if not (code_str.startswith("<") and ">" in code_str):
#                 raise ValueError("Invalid XML format")
                
#             # 如果没有单一根节点，添加一个root标签
#             if not (code_str.strip().startswith("<?xml") or 
#                    (code_str.strip().startswith("<") and code_str.strip().endswith(">") and 
#                     code_str.count("<") - code_str.count("</") == 1)):
#                 code_str = f"<root>{code_str}</root>"
                
#             parsed = xmltodict.parse(code_str)
#         else:
#             # 如果格式未知，尝试自动检测
#             for parser_fmt, parser_func in [
#                 ("json", json.loads),
#                 ("json5", json5.loads),
#                 ("hjson", hjson.loads),
#                 ("yaml", yaml.safe_load),
#                 ("toml", toml.loads)
#             ]:
#                 try:
#                     parsed = parser_func(code_str)
#                     print(f"自动检测格式为: {parser_fmt}")
#                     break
#                 except Exception:
#                     continue
#             else:
#                 # 如果所有格式都失败了
#                 raise ValueError(f"无法解析未知格式: {fmt}")
        
#         # 统一转换为排序的JSON字符串
#         return json.dumps(parsed, indent=2, sort_keys=True)
#     except Exception as e:
#         raise ValueError(f"转换格式 {fmt} 到 JSON 失败: {str(e)}\n代码片段: {code_str[:100]}...")

# def compare_structures_tree_edit_distance(code1, code2, fmt1="json", fmt2=None):
#     """
#     比较两个不同格式代码的结构相似度
    
#     Args:
#         code1: 第一个代码字符串（参考/正确答案）
#         code2: 第二个代码字符串（用户答案）
#         fmt1: 第一个代码的格式
#         fmt2: 第二个代码的格式，如果为None则与fmt1相同
    
#     Returns:
#         包含树编辑距离、归一化距离和相似度分数的字典
#     """
#     if fmt2 is None:
#         fmt2 = fmt1
    
#     try:
#         json1 = convert_to_json_string(code1, fmt1)
#         json2 = convert_to_json_string(code2, fmt2)
#     except Exception as e:
#         return {
#             "error": f"解析失败 ({fmt1}/{fmt2}): {str(e)}",
#             "ted": None,
#             "normalized_ted": None,
#             "similarity_score": None
#         }

#     source1 = bytes(json1, "utf-8")
#     source2 = bytes(json2, "utf-8")

#     try:
#         tree1 = parser.parse(source1)
#         tree2 = parser.parse(source2)
#     except Exception as e:
#         return {"error": f"Tree解析失败: {e}"}

#     zss_tree1 = NodeWrapper(tree1.root_node, source1)
#     zss_tree2 = NodeWrapper(tree2.root_node, source2)

#     try:
#         ted = zss.simple_distance(
#             zss_tree1, zss_tree2,
#             get_children=lambda node: node.get_children(),
#             get_label=lambda node: node.get_label()
#         )

#         size1 = tree_size(zss_tree1)
#         size2 = tree_size(zss_tree2)
#         max_size = max(size1, size2)
#         n_ted = ted / max_size if max_size else 0
#         similarity = 1 - n_ted

#         return {
#             "ted": ted,
#             "normalized_ted": round(n_ted, 4),
#             "similarity_score": round(similarity, 4),
#             "tree_size1": size1,
#             "tree_size2": size2,
#             "error": None
#         }
#     except Exception as e:
#         return {"error": f"计算树编辑距离失败: {e}"}

# def detect_format(code_str):
#     """尝试自动检测代码格式"""
#     # 检查是否是XML
#     if code_str.strip().startswith("<") and ">" in code_str:
#         return "xml"
    
#     # 检查是否包含 TOML 标识符
#     if re.search(r"^\[.*\]", code_str, re.MULTILINE):
#         return "toml"
    
#     # 检查缩进和使用的分隔符来区分 YAML 和 JSON 系列
#     if ":" in code_str and ("{" not in code_str or "[" not in code_str):
#         return "yaml"
    
#     # 检查是否包含 JSON5 特性
#     if re.search(r"//|/\*|\*/|,$", code_str):
#         return "json5"
    
#     # 默认为标准 JSON
#     return "json"

# def batch_tree_distance_analysis(all_data, gold_struct_code, gold_format="json"):
#     """
#     批量比较用户答案与标准答案的结构相似度
    
#     Args:
#         all_data: 包含所有用户答案的数据字典
#         gold_struct_code: 标准答案代码字符串
#         gold_format: 标准答案的格式
    
#     Returns:
#         包含比较结果的DataFrame
#     """
#     results = []

#     for file_name, quiz_data in all_data.items():
#         answers = quiz_data.get('answers', {})
        
#         # 提取 participantId
#         participant_id = file_name
#         for v in answers.values():
#             if isinstance(v, dict):
#                 a = v.get("answer", {})
#                 if isinstance(a, dict) and "prolificId" in a:
#                     participant_id = a["prolificId"]
#                     break

#         # 推测用户使用的格式
#         format_name = None
#         # 从教程部分推测格式
#         for k in answers:
#             m = re.match(r"tutorial-(\w+)-part1", k)
#             if m:
#                 format_name = m.group(1).lower()
#                 break
#         # 从写作任务键名推测格式
#         if not format_name:
#             for k in answers:
#                 if "writing-task-tabular-" in k:
#                     parts = k.replace("writing-task-tabular-", "").split("_")
#                     if parts:
#                         format_name = parts[0].lower()
#                         break
#         # 默认未知格式
#         if not format_name:
#             format_name = "unknown"

#         # 处理每个写作任务
#         for key, content in answers.items():
#             if key.startswith("writing-task-tabular-") and "_post-task-question" not in key and isinstance(content, dict):
#                 ans_block = content.get("answer", {})
#                 code = ans_block.get("code") if isinstance(ans_block, dict) else None
                
#                 if not code:
#                     continue
                
#                 # 如果格式仍然未知，尝试自动检测
#                 user_format = format_name
#                 if user_format == "unknown":
#                     try:
#                         user_format = detect_format(code)
#                     except:
#                         pass

#                 # Tree Edit Distance 分析
#                 try:
#                     ted_result = compare_structures_tree_edit_distance(
#                         gold_struct_code, code, 
#                         fmt1=gold_format, 
#                         fmt2=user_format
#                     )
                    
#                     ted = ted_result.get("ted")
#                     norm_ted = ted_result.get("normalized_ted")
#                     sim = ted_result.get("similarity_score")
#                     size1 = ted_result.get("tree_size1")
#                     size2 = ted_result.get("tree_size2")
#                     err = ted_result.get("error", "")
#                 except Exception as e:
#                     ted = None
#                     norm_ted = None
#                     sim = None
#                     size1 = None
#                     size2 = None
#                     err = f"TED错误: {e}"

#                 results.append({
#                     "participantId": participant_id,
#                     "format": user_format,
#                     "task": key,
#                     "ted": ted,
#                     "normalized_ted": norm_ted,
#                     "similarity_score": sim,
#                     "gold_tree_size": size1,
#                     "user_tree_size": size2,
#                     "error": err
#                 })

#     return pd.DataFrame(results)


# def batch_tree_distance_analysis_tree_only(all_data, gold_struct_data):
#     """
#     通过树编辑距离批量比较用户答案与标准答案的结构相似度
    
#     Args:
#         all_data: 包含所有用户答案的数据字典
#         gold_struct_data: 标准答案的Python字典对象 (不是字符串)
    
#     Returns:
#         包含比较结果的DataFrame
#     """
#     # 将Python字典转换为JSON字符串，用于树结构比较
#     gold_struct_code = json.dumps(gold_struct_data, indent=2, sort_keys=True)
    
#     # 调用原有比较函数
#     return batch_tree_distance_analysis(all_data, gold_struct_code, gold_format="json")

# def evaluate_user_solution(user_code, gold_struct_code, user_format=None):
#     """
#     单独评估一个用户解决方案与标准答案的相似度
    
#     Args:
#         user_code: 用户的代码字符串
#         gold_struct_code: 标准答案代码字符串
#         user_format: 用户代码的格式，如果为None则自动检测
    
#     Returns:
#         包含比较结果的字典
#     """
#     if user_format is None:
#         user_format = detect_format(user_code)
    
#     result = compare_structures_tree_edit_distance(
#         gold_struct_code, user_code, 
#         fmt1="json", 
#         fmt2=user_format
#     )
    
#     return {
#         "format": user_format,
#         "similarity_score": result.get("similarity_score"),
#         "normalized_ted": result.get("normalized_ted"),
#         "tree_edit_distance": result.get("ted"),
#         "gold_tree_size": result.get("tree_size1"),
#         "user_tree_size": result.get("tree_size2"),
#         "error": result.get("error")
#     }


# 设置 Tree-sitter JSON parser（请确保 build/my-languages.so 已构建）
try:
    JSON_LANGUAGE = Language('build/my-languages.so', 'json')
    parser = Parser()
    parser.set_language(JSON_LANGUAGE)
except Exception as e:
    print(f"初始化Tree-sitter解析器失败: {e}")
    print("请确保已编译tree-sitter语言库到build/my-languages.so")

# Tree node wrapper for ZSS
class NodeWrapper:
    def __init__(self, node, source):
        self.node = node
        self.source = source

    def get_label(self):
        try:
            return f"{self.node.type}:{self.source[self.node.start_byte:self.node.end_byte].decode('utf-8').strip()}"
        except Exception:
            return f"{self.node.type}:error"

    def get_children(self):
        return [NodeWrapper(child, self.source) for child in self.node.children]

def tree_size(node):
    """计算树的节点数量"""
    return 1 + sum(tree_size(child) for child in node.get_children())

def clean_comments(code_str):
    """清除代码中的注释"""
    # 清除单行注释
    code_str = re.sub(r"//.*?$", "", code_str, flags=re.MULTILINE)
    # 清除多行注释
    code_str = re.sub(r"/\*.*?\*/", "", code_str, flags=re.DOTALL)
    return code_str

def convert_to_json_string(code_str, fmt):
    """转换任意支持格式为标准 JSON 字符串（供 tree-sitter 使用）"""
    if not code_str or not isinstance(code_str, str):
        raise ValueError(f"Invalid code string: {type(code_str)}")
    
    code_str = code_str.strip()
    fmt = fmt.lower()
    
    # 预处理：删除前后空白字符，处理缩进问题
    lines = code_str.split('\n')
    # 跳过空行
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise ValueError("Empty code string")
        
    # 修复可能的缩进问题（对YAML尤其重要）
    if fmt == "yaml" or fmt == "yml":
        # 检查是否有适当的缩进
        has_indent = any(line.startswith(' ') or line.startswith('\t') for line in lines[1:])
        if not has_indent and len(lines) > 1:
            # 没有缩进，尝试添加标准缩进
            for i in range(1, len(lines)):
                if not lines[i].startswith(' ') and not lines[i].startswith('-'):
                    lines[i] = '  ' + lines[i]
        code_str = '\n'.join(lines)
    
    try:
        if fmt == "json":
            # 尝试处理格式不规范的JSON
            try:
                parsed = json.loads(code_str)
            except json.JSONDecodeError as e:
                # 修复常见JSON格式问题
                fixed_code = code_str
                # 1. 尝试添加丢失的引号
                fixed_code = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', fixed_code)
                # 2. 尝试修复尾随逗号
                fixed_code = re.sub(r',(\s*[}\]])', r'\1', fixed_code)
                try:
                    parsed = json.loads(fixed_code)
                except Exception:
                    # 如果仍然失败，尝试hjson解析
                    try:
                        parsed = hjson.loads(code_str)
                    except Exception as hjson_err:
                        raise ValueError(f"无法解析JSON: {e}, 尝试hjson也失败: {hjson_err}")
        elif fmt == "jsonc":
            cleaned_code = clean_comments(code_str)
            try:
                parsed = json.loads(cleaned_code)
            except Exception as json_err:
                # 如果失败，尝试hjson
                try:
                    parsed = hjson.loads(cleaned_code)
                except Exception:
                    raise json_err
        elif fmt == "json5":
            try:
                parsed = json5.loads(code_str)
            except Exception as e:
                # 尝试修复常见JSON5问题
                fixed_code = code_str
                # 修复注释
                fixed_code = clean_comments(fixed_code)
                # 尝试hjson作为备选
                try:
                    parsed = hjson.loads(fixed_code)
                except Exception:
                    raise e
        elif fmt == "hjson":
            parsed = hjson.loads(code_str)
        elif fmt == "yaml" or fmt == "yml":
            try:
                # 安全加载，添加安全实用选项
                parsed = yaml.safe_load(code_str)
                if parsed is None:
                    # 有时候yaml.safe_load返回None，尝试不同的方法
                    parsed = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            parsed[key.strip()] = value.strip() if value.strip() else None
            except Exception as e:
                # 尝试修复常见YAML问题
                fixed_yaml = ""
                indent_level = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped.endswith(':'):
                        fixed_yaml += ' ' * indent_level + stripped + '\n'
                        indent_level += 2
                    elif stripped.startswith('-'):
                        fixed_yaml += ' ' * indent_level + stripped + '\n'
                    else:
                        fixed_yaml += ' ' * indent_level + stripped + '\n'
                
                try:
                    parsed = yaml.safe_load(fixed_yaml)
                    if parsed is None:
                        raise ValueError("YAML解析为None")
                except Exception:
                    # 最后尝试将YAML视为简单的键值对
                    parsed = {}
                    for line in lines:
                        if ':' in line and not line.strip().startswith('#'):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip()
                                value = parts[1].strip()
                                # 处理嵌套
                                if not key.startswith(' ') and not key.startswith('\t'):
                                    parsed[key] = value if value else {}
                    
                    if not parsed:
                        raise ValueError(f"无法解析YAML: {e}")
        elif fmt == "toml":
            try:
                parsed = toml.loads(code_str)
            except Exception as e:
                # 尝试修复TOML格式问题
                fixed_toml = ""
                in_section = False
                section_name = ""
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('[') and stripped.endswith(']'):
                        in_section = True
                        section_name = stripped[1:-1]
                        fixed_toml += stripped + '\n'
                    elif '=' in stripped:
                        fixed_toml += stripped + '\n'
                    elif in_section and stripped and not stripped.startswith('#'):
                        # 可能是缺少键值对分隔符的行
                        fixed_toml += section_name + '.' + stripped + ' = ""\n'
                
                try:
                    parsed = toml.loads(fixed_toml)
                except Exception:
                    raise e
        elif fmt == "xml":
            # 确保有根元素
            if not (code_str.startswith("<") and ">" in code_str):
                raise ValueError("Invalid XML format")
                
            # 如果没有单一根节点，添加一个root标签
            if not (code_str.strip().startswith("<?xml") or 
                   (code_str.strip().startswith("<") and code_str.strip().endswith(">") and 
                    code_str.count("<") - code_str.count("</") == 1)):
                code_str = f"<root>{code_str}</root>"
            
            try:
                parsed = xmltodict.parse(code_str)
            except Exception as e:
                # 尝试修复常见的XML错误
                fixed_xml = code_str
                # 1. 添加缺失的结束标签
                open_tags = re.findall(r'<(\w+)[^>]*>', fixed_xml)
                close_tags = re.findall(r'</(\w+)>', fixed_xml)
                
                for tag in open_tags:
                    if tag not in close_tags and not f"<{tag}/>" in fixed_xml:
                        fixed_xml += f"</{tag}>"
                
                # 2. 尝试封闭自闭合标签
                fixed_xml = re.sub(r'<(\w+)([^>]*)>(\s*)</\1>', r'<\1\2/>', fixed_xml)
                
                try:
                    parsed = xmltodict.parse(fixed_xml)
                except Exception:
                    raise e
        else:
            # 如果格式未知，尝试自动检测
            for parser_fmt, parser_func in [
                ("json", json.loads),
                ("json5", json5.loads),
                ("hjson", hjson.loads),
                ("yaml", yaml.safe_load),
                ("toml", toml.loads)
            ]:
                try:
                    parsed = parser_func(code_str)
                    print(f"自动检测格式为: {parser_fmt}")
                    break
                except Exception:
                    continue
            else:
                # 最后尝试作为简单文本解析
                lines_dict = {}
                for i, line in enumerate(lines):
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            lines_dict[parts[0].strip()] = parts[1].strip()
                    else:
                        lines_dict[f"line_{i}"] = line.strip()
                
                if lines_dict:
                    parsed = lines_dict
                else:
                    raise ValueError(f"无法解析未知格式: {fmt}")
        
        # 统一转换为排序的JSON字符串
        return json.dumps(parsed, indent=2, sort_keys=True)
    except Exception as e:
        raise ValueError(f"转换格式 {fmt} 到 JSON 失败: {str(e)}\n代码片段: {code_str[:100]}...")


def compare_structures_direct(data1, data2):
    """
    直接比较两个数据结构（不经过树编辑距离），作为备选方案
    
    Args:
        data1: 第一个数据结构（字典或列表）
        data2: 第二个数据结构（字典或列表）
    
    Returns:
        包含相似度分数的字典
    """
    def flatten_dict(d, parent_key=''):
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
        elif isinstance(d, list):
            for i, v in enumerate(d):
                new_key = f"{parent_key}[{i}]"
                if isinstance(v, (dict, list)):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
        return dict(items)
    
    try:
        # 将两个结构扁平化为键值对
        flat1 = flatten_dict(data1)
        flat2 = flatten_dict(data2)
        
        # 计算键的交集和并集
        keys1 = set(flat1.keys())
        keys2 = set(flat2.keys())
        
        keys_intersection = keys1.intersection(keys2)
        keys_union = keys1.union(keys2)
        
        # 计算键的Jaccard相似度
        key_similarity = len(keys_intersection) / len(keys_union) if keys_union else 1.0
        
        # 计算值的相似度（对于共同的键）
        value_matches = sum(1 for k in keys_intersection if flat1[k] == flat2[k])
        value_similarity = value_matches / len(keys_intersection) if keys_intersection else 1.0
        
        # 综合分数 (70% 结构, 30% 值)
        combined_score = 0.7 * key_similarity + 0.3 * value_similarity
        
        return {
            "similarity_score": round(combined_score, 4),
            "key_similarity": round(key_similarity, 4),
            "value_similarity": round(value_similarity, 4),
            "common_keys": len(keys_intersection),
            "total_keys": len(keys_union)
        }
    except Exception as e:
        return {
            "error": f"直接比较失败: {e}",
            "similarity_score": 0
        }


def detect_format(code_str):
    """尝试自动检测代码格式"""
    # 检查是否是XML
    if code_str.strip().startswith("<") and ">" in code_str:
        return "xml"
    
    # 检查是否包含 TOML 标识符
    if re.search(r"^\[.*\]", code_str, re.MULTILINE):
        return "toml"
    
    # 检查缩进和使用的分隔符来区分 YAML 和 JSON 系列
    if ":" in code_str and ("{" not in code_str or "[" not in code_str):
        return "yaml"
    
    # 检查是否包含 JSON5 特性
    if re.search(r"//|/\*|\*/|,$", code_str):
        return "json5"
    
    # 默认为标准 JSON
    return "json"


def batch_tree_distance_analysis_tree_only(all_data, gold_struct_data):
    """
    通过树编辑距离批量比较用户答案与标准答案的结构相似度
    
    Args:
        all_data: 包含所有用户答案的数据字典
        gold_struct_data: 标准答案的Python字典对象 (不是字符串)
    
    Returns:
        包含比较结果的DataFrame
    """
    # 将Python字典转换为JSON字符串，用于树结构比较
    gold_struct_code = json.dumps(gold_struct_data, indent=2, sort_keys=True)
    
    # 调用原有比较函数
    return batch_tree_distance_analysis(all_data, gold_struct_code, gold_format="json")

def evaluate_user_solution(user_code, gold_struct_code, user_format=None):
    """
    单独评估一个用户解决方案与标准答案的相似度
    
    Args:
        user_code: 用户的代码字符串
        gold_struct_code: 标准答案代码字符串
        user_format: 用户代码的格式，如果为None则自动检测
    
    Returns:
        包含比较结果的字典
    """
    if user_format is None:
        user_format = detect_format(user_code)
    
    result = compare_structures_tree_edit_distance(
        gold_struct_code, user_code, 
        fmt1="json", 
        fmt2=user_format
    )
    
    return {
        "format": user_format,
        "similarity_score": result.get("similarity_score"),
        "normalized_ted": result.get("normalized_ted"),
        "tree_edit_distance": result.get("ted"),
        "gold_tree_size": result.get("tree_size1"),
        "user_tree_size": result.get("tree_size2"),
        "error": result.get("error")
    }


    """
    批量比较用户答案与标准答案的结构相似度
    
    Args:
        all_data: 包含所有用户答案的数据字典
        gold_struct_code: 标准答案代码字符串
        gold_format: 标准答案的格式
    
    Returns:
        包含比较结果的DataFrame
    """
    results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})
        
        # 提取 participantId
        participant_id = file_name
        for v in answers.values():
            if isinstance(v, dict):
                a = v.get("answer", {})
                if isinstance(a, dict) and "prolificId" in a:
                    participant_id = a["prolificId"]
                    break

        # 推测用户使用的格式
        format_name = None
        # 从教程部分推测格式
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break
        # 从写作任务键名推测格式
        if not format_name:
            for k in answers:
                if "writing-task-tabular-" in k:
                    parts = k.replace("writing-task-tabular-", "").split("_")
                    if parts:
                        format_name = parts[0].lower()
                        break
        # 默认未知格式
        if not format_name:
            format_name = "unknown"

        # 处理每个写作任务
        for key, content in answers.items():
            if key.startswith("writing-task-tabular-") and "_post-task-question" not in key and isinstance(content, dict):
                ans_block = content.get("answer", {})
                code = ans_block.get("code") if isinstance(ans_block, dict) else None
                
                if not code:
                    continue
                
                # 如果格式仍然未知，尝试自动检测
                user_format = format_name
                if user_format == "unknown":
                    try:
                        user_format = detect_format(code)
                    except:
                        pass

                # Tree Edit Distance 分析
                try:
                    ted_result = compare_structures_tree_edit_distance(
                        gold_struct_code, code, 
                        fmt1=gold_format, 
                        fmt2=user_format
                    )
                    
                    ted = ted_result.get("ted")
                    norm_ted = ted_result.get("normalized_ted")
                    sim = ted_result.get("similarity_score")
                    size1 = ted_result.get("tree_size1")
                    size2 = ted_result.get("tree_size2")
                    err = ted_result.get("error", "")
                except Exception as e:
                    # 如果分析完全失败，将相似度设为0
                    ted = None
                    norm_ted = 1.0  # 最大归一化距离
                    sim = 0.0  # 最小相似度
                    size1 = None
                    size2 = None
                    err = f"TED错误: {e}"

                results.append({
                    "participantId": participant_id,
                    "format": user_format,
                    "task": key,
                    "ted": ted,
                    "normalized_ted": norm_ted,
                    "similarity_score": sim,
                    "gold_tree_size": size1,
                    "user_tree_size": size2,
                    "error": err
                })

    return pd.DataFrame(results)



def compare_structures_tree_edit_distance(code1, code2, fmt1="json", fmt2=None):
    """
    比较两个不同格式代码的结构相似度
    
    Args:
        code1: 第一个代码字符串（参考/正确答案）
        code2: 第二个代码字符串（用户答案）
        fmt1: 第一个代码的格式
        fmt2: 第二个代码的格式，如果为None则与fmt1相同
    
    Returns:
        包含树编辑距离、归一化距离和相似度分数的字典
    """
    if fmt2 is None:
        fmt2 = fmt1
    
    # 步骤1: 尝试解析两个代码为JSON对象
    try:
        # 先尝试将代码解析为Python对象
        if fmt1 == "json":
            data1 = json.loads(code1)
        elif fmt1 == "yaml" or fmt1 == "yml":
            data1 = yaml.safe_load(code1)
        elif fmt1 == "toml":
            data1 = toml.loads(code1)
        elif fmt1 == "xml":
            if not code1.strip().startswith("<root>"):
                code1 = f"<root>{code1}</root>"
            data1 = xmltodict.parse(code1)
        else:
            # 尝试使用适当函数
            data1 = eval(f"{fmt1}.loads(code1)")
        
        if fmt2 == "json":
            data2 = json.loads(code2)
        elif fmt2 == "yaml" or fmt2 == "yml":
            data2 = yaml.safe_load(code2)
        elif fmt2 == "toml":
            data2 = toml.loads(code2)
        elif fmt2 == "xml":
            if not code2.strip().startswith("<root>"):
                code2 = f"<root>{code2}</root>"
            data2 = xmltodict.parse(code2)
        else:
            # 尝试使用适当函数
            data2 = eval(f"{fmt2}.loads(code2)")
            
        # 如果能成功解析为Python对象，先尝试直接比较
        direct_result = compare_structures_direct(data1, data2)
        
        # 如果直接比较成功并且得分很高，可以直接返回
        if direct_result.get("error") is None and direct_result.get("similarity_score", 0) > 0.9:
            direct_result["method"] = "direct_comparison"
            return direct_result
    except Exception:
        # 如果解析失败，继续使用树编辑距离
        pass
    
    # 步骤2: 尝试转换为JSON字符串并使用树编辑距离
    try:
        json1 = convert_to_json_string(code1, fmt1)
        json2 = convert_to_json_string(code2, fmt2)
    except Exception as e:
        # 如果转换失败，但之前的直接比较成功了，返回直接比较结果
        if 'direct_result' in locals() and direct_result.get("error") is None:
            direct_result["method"] = "direct_comparison"
            direct_result["ted_error"] = str(e)
            return direct_result
        # 如果无法解析为树，则相似度为0
        return {
            "error": f"解析失败 ({fmt1}/{fmt2}): {str(e)}",
            "ted": None,
            "normalized_ted": 1.0,  # 最大归一化距离
            "similarity_score": 0.0  # 最小相似度
        }

    source1 = bytes(json1, "utf-8")
    source2 = bytes(json2, "utf-8")

    try:
        tree1 = parser.parse(source1)
        tree2 = parser.parse(source2)
    except Exception as e:
        # 如果树解析失败，但之前的直接比较成功了，返回直接比较结果
        if 'direct_result' in locals() and direct_result.get("error") is None:
            direct_result["method"] = "direct_comparison"
            direct_result["tree_error"] = str(e)
            return direct_result
        # 如果无法解析为树，则相似度为0
        return {
            "error": f"Tree解析失败: {e}",
            "ted": None,
            "normalized_ted": 1.0,
            "similarity_score": 0.0
        }

    zss_tree1 = NodeWrapper(tree1.root_node, source1)
    zss_tree2 = NodeWrapper(tree2.root_node, source2)

    try:
        ted = zss.simple_distance(
            zss_tree1, zss_tree2,
            get_children=lambda node: node.get_children(),
            get_label=lambda node: node.get_label()
        )

        size1 = tree_size(zss_tree1)
        size2 = tree_size(zss_tree2)
        max_size = max(size1, size2)
        n_ted = ted / max_size if max_size else 0
        similarity = 1 - n_ted

        result = {
            "ted": ted,
            "normalized_ted": round(n_ted, 4),
            "similarity_score": round(similarity, 4),
            "tree_size1": size1,
            "tree_size2": size2,
            "error": None,
            "method": "tree_edit_distance"
        }
        
        # 如果有直接比较结果，合并两种方法的分数
        if 'direct_result' in locals() and direct_result.get("error") is None:
            result["direct_similarity"] = direct_result.get("similarity_score")
            # 使用加权平均
            result["similarity_score"] = round(
                (result["similarity_score"] + direct_result.get("similarity_score", 0)) / 2,
                4
            )
        
        return result
    except Exception as e:
        # 如果树编辑距离计算失败，但之前的直接比较成功了，返回直接比较结果
        if 'direct_result' in locals() and direct_result.get("error") is None:
            direct_result["method"] = "direct_comparison"
            direct_result["ted_error"] = str(e)
            return direct_result
        # 如果无法计算树编辑距离，则相似度为0
        return {
            "error": f"计算树编辑距离失败: {e}",
            "ted": None,
            "normalized_ted": 1.0,
            "similarity_score": 0.0
        }


def calculate_format_similarity_stats(results_df):
    """
    计算每种格式的平均相似度统计信息
    
    Args:
        results_df: 包含比较结果的DataFrame
    
    Returns:
        包含每种格式统计信息的DataFrame
    """
    # 确保similarity_score列中的None值被替换为0.0
    results_df['similarity_score'] = results_df['similarity_score'].fillna(0.0)
    
    # 按格式分组并计算统计信息
    format_stats = results_df.groupby('format').agg({
        'similarity_score': ['mean', 'std', 'count', 'min', 'max'],
        'normalized_ted': ['mean', 'std', 'min', 'max'],
        'error': lambda x: ((x.notna()) & (x != '') & (x != 'None')).sum()
        # 'error': lambda x: sum(x != '')  # 计算有错误的记录数
    })
    
    # 展平多级索引
    format_stats.columns = ['_'.join(col).strip() for col in format_stats.columns.values]
    
    # 计算成功率（没有错误的比例）
    format_stats['success_rate'] = 1 - (format_stats['error_<lambda>'] / format_stats['similarity_score_count'])
    
    # 重新排序列
    format_stats = format_stats.sort_values('similarity_score_mean', ascending=False)
    
    return format_stats


def visualize_format_similarity(format_stats, output_file=None):
    """
    可视化每种格式的平均相似度
    
    Args:
        format_stats: 包含每种格式统计信息的DataFrame
        output_file: 输出文件路径，如果为None则显示图表
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 设置样式
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    formats = format_stats.index
    mean_scores = format_stats['similarity_score_mean']
    std_scores = format_stats['similarity_score_std']
    counts = format_stats['similarity_score_count']
    
    # 创建条形图
    ax = sns.barplot(x=formats, y=mean_scores, palette="viridis")
    
    # 添加误差线
    ax.errorbar(
        x=range(len(formats)),
        y=mean_scores,
        yerr=std_scores,
        fmt='none',
        ecolor='black',
        capsize=5
    )
    
    # 添加样本数量标签
    for i, count in enumerate(counts):
        ax.text(i, 0.05, f"n={count}", ha='center', va='bottom', color='white', fontweight='bold')
    
    # 设置标题和标签
    plt.title('average similarity by format', fontsize=16)
    plt.xlabel('format', fontsize=14)
    plt.ylabel('average similarity', fontsize=14)
    
    # 设置y轴范围
    plt.ylim(0, 1.0)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加均值线
    plt.axhline(y=mean_scores.mean(), color='r', linestyle='--', alpha=0.7)
    plt.text(
        len(formats) - 1, 
        mean_scores.mean() + 0.02, 
        f'total average: {mean_scores.mean():.3f}', 
        color='r'
    )
    
    # 添加数据标签
    for i, score in enumerate(mean_scores):
        ax.text(i, score + 0.02, f'{score:.3f}', ha='center', va='bottom')
    
    # 紧凑布局
    plt.tight_layout()
    plt.show()
    
    # 创建成功率图表
    plt.figure(figsize=(12, 6))
    success_rates = format_stats['success_rate']
    
    ax = sns.barplot(x=formats, y=success_rates, palette="coolwarm")
    
    plt.title('format parse success rate', fontsize=16)
    plt.xlabel('format', fontsize=14)
    plt.ylabel('success rate', fontsize=14)
    plt.ylim(0, 1.0)
    
    # 添加数据标签
    for i, rate in enumerate(success_rates):
        ax.text(i, rate + 0.02, f'{rate:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
        
    # 创建样本数量分布图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=formats, y=counts, palette="muted")
    
    plt.title('Distribution of sample quantity in each format', fontsize=16)
    plt.xlabel('format', fontsize=14)
    plt.ylabel('sampe', fontsize=14)
    
    # 添加数据标签
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(int(count)), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    


def batch_tree_distance_analysis(all_data, gold_struct_code, gold_format="json"):
    """
    批量比较用户答案与标准答案的结构相似度
    
    Args:
        all_data: 包含所有用户答案的数据字典
        gold_struct_code: 标准答案代码字符串
        gold_format: 标准答案的格式
    
    Returns:
        包含比较结果的DataFrame
    """
    results = []

    for file_name, quiz_data in all_data.items():
        answers = quiz_data.get('answers', {})
        
        # 提取 participantId
        participant_id = file_name
        for v in answers.values():
            if isinstance(v, dict):
                a = v.get("answer", {})
                if isinstance(a, dict) and "prolificId" in a:
                    participant_id = a["prolificId"]
                    break

        # 推测用户使用的格式
        format_name = None
        # 从教程部分推测格式
        for k in answers:
            m = re.match(r"tutorial-(\w+)-part1", k)
            if m:
                format_name = m.group(1).lower()
                break
        # 从写作任务键名推测格式
        if not format_name:
            for k in answers:
                if "writing-task-tabular-" in k:
                    parts = k.replace("writing-task-tabular-", "").split("_")
                    if parts:
                        format_name = parts[0].lower()
                        break
        # 默认未知格式
        if not format_name:
            format_name = "unknown"

        # 处理每个写作任务
        for key, content in answers.items():
            if key.startswith("writing-task-tabular-") and "_post-task-question" not in key and isinstance(content, dict):
                ans_block = content.get("answer", {})
                code = ans_block.get("code") if isinstance(ans_block, dict) else None
                
                if not code:
                    continue
                
                # 如果格式仍然未知，尝试自动检测
                user_format = format_name
                if user_format == "unknown":
                    try:
                        user_format = detect_format(code)
                    except:
                        pass

                # Tree Edit Distance 分析
                try:
                    ted_result = compare_structures_tree_edit_distance(
                        gold_struct_code, code, 
                        fmt1=gold_format, 
                        fmt2=user_format
                    )
                    
                    ted = ted_result.get("ted")
                    norm_ted = ted_result.get("normalized_ted")
                    sim = ted_result.get("similarity_score")
                    size1 = ted_result.get("tree_size1")
                    size2 = ted_result.get("tree_size2")
                    err = ted_result.get("error", "")
                except Exception as e:
                    # 如果分析完全失败，将相似度设为0
                    ted = None
                    norm_ted = 1.0  # 最大归一化距离
                    sim = 0.0  # 最小相似度
                    size1 = None
                    size2 = None
                    err = f"TED错误: {e}"

                results.append({
                    "participantId": participant_id,
                    "format": user_format,
                    "task": key,
                    "ted": ted,
                    "normalized_ted": norm_ted,
                    "similarity_score": sim,
                    "gold_tree_size": size1,
                    "user_tree_size": size2,
                    "error": err
                })

    return pd.DataFrame(results)


# def analyze_and_visualize_results(all_data, gold_struct_code, gold_format="json", output_file=None):
#     """
#     一站式分析和可视化结果
    
#     Args:
#         all_data: 包含所有用户答案的数据字典
#         gold_struct_code: 标准答案代码字符串
#         gold_format: 标准答案的格式
#         output_file: 可视化输出文件，如果为None则显示图表
    
#     Returns:
#         包含比较结果的DataFrame和统计信息
#     """
#     # 进行批量分析
#     results_df = batch_tree_distance_analysis(all_data, gold_struct_code, gold_format)
    
#     # 计算每种格式的统计信息
#     format_stats = calculate_format_similarity_stats(results_df)
    
#     # 可视化结果
#     visualize_format_similarity(format_stats, output_file)
    
#     return results_df, format_stats




    
    """
    一站式分析和可视化结果
    
    Args:
        all_data: 包含所有用户答案的数据字典
        gold_struct_code: 标准答案代码字符串
        gold_format: 标准答案的格式
        output_file: 可视化输出文件，如果为None则显示图表
    
    Returns:
        包含比较结果的DataFrame和统计信息
    """
    # 进行批量分析
    results_df = batch_tree_distance_analysis(all_data, gold_struct_code, gold_format)
    
    # 计算每种格式的统计信息
    format_stats = calculate_format_similarity_stats(results_df)
    
    # 可视化结果
    visualize_format_similarity(format_stats, output_file)
    
    return results_df, format_stats