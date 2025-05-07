from tree_sitter import Language

# 指定输出路径和语法仓库路径
Language.build_library(
  # 输出文件
  'build/my-languages.so',
  # 语法仓库目录列表
  [
    'tree-sitter-json'   # 相对路径，也可以用绝对路径
  ]
)
print("✅ build/my-languages.so created")
