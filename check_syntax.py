#!/usr/bin/env python3
"""
简单的语法检查脚本
"""

import ast
import sys

def check_syntax(filename):
    """检查Python文件的语法"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # 尝试解析AST
        ast.parse(source, filename=filename)
        print(f"✅ {filename} 语法正确")
        return True
        
    except SyntaxError as e:
        print(f"❌ {filename} 语法错误:")
        print(f"   行 {e.lineno}: {e.text.strip() if e.text else ''}")
        print(f"   错误: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {filename} 检查失败: {str(e)}")
        return False

if __name__ == "__main__":
    filename = "src/business/langgraph_form_processor.py"
    check_syntax(filename) 