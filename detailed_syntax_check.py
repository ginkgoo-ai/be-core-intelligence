#!/usr/bin/env python3
"""
详细的语法检查脚本
"""

import ast
import sys

def check_syntax_detailed(filename):
    """详细检查Python文件的语法"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 逐行检查，找到问题行
        for i, line in enumerate(lines, 1):
            if i >= 800 and i <= 810:
                print(f"行 {i:3d}: {repr(line)}")
        
        # 尝试解析整个文件
        source = ''.join(lines)
        ast.parse(source, filename=filename)
        print(f"✅ {filename} 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} 语法错误:")
        print(f"   行 {e.lineno}: {repr(e.text) if e.text else 'None'}")
        print(f"   错误: {e.msg}")
        print(f"   位置: {e.offset}")
        
        # 显示错误行周围的上下文
        if e.lineno:
            start = max(1, e.lineno - 3)
            end = min(len(lines), e.lineno + 3)
            print(f"\n上下文 (行 {start}-{end}):")
            for i in range(start-1, end):
                marker = ">>> " if i+1 == e.lineno else "    "
                print(f"{marker}{i+1:3d}: {repr(lines[i])}")
        
        return False
    except Exception as e:
        print(f"❌ {filename} 检查失败: {str(e)}")
        return False

if __name__ == "__main__":
    filename = "src/business/langgraph_form_processor.py"
    check_syntax_detailed(filename) 