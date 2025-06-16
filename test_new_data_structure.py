#!/usr/bin/env python3
"""
测试新的数据结构格式
验证 LangGraphFormProcessor 是否正确保存数据到 step_instance
"""

import json
import os
import sys
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.database_config import get_db_session
from src.business.langgraph_form_processor import LangGraphFormProcessor
from src.database.workflow_repositories import StepInstanceRepository

def test_data_structure():
    """测试新的数据结构格式"""
    
    # 测试HTML表单
    test_html = """
    <form>
        <div>
            <label for="visaType">选择您的签证类型</label>
            <select id="visaType" name="visaType" required>
                <option value="">请选择</option>
                <option value="visit-visa-ooc-standard">Visit or transit visa</option>
                <option value="tier-1-investor-ooc">Tier 1 (Investor)</option>
                <option value="skilled-worker-ooc">Skilled Worker visa</option>
            </select>
        </div>
        
        <div>
            <label for="firstName">名字</label>
            <input type="text" id="firstName" name="firstName" required>
        </div>
        
        <div>
            <label for="email">邮箱</label>
            <input type="email" id="email" name="email" required>
        </div>
        
        <input type="hidden" name="csrfToken" value="abc123">
        
        <button type="submit">提交</button>
    </form>
    """
    
    # 测试用户资料数据
    test_profile = {
        "personal_info": {
            "first_name": "张",
            "last_name": "三",
            "email": "zhangsan@example.com"
        },
        "visa_preferences": {
            "visa_type": "visit-visa-ooc-standard",
            "purpose": "tourism"
        }
    }
    
    # 创建数据库会话
    db = next(get_db_session())
    
    try:
        # 创建处理器
        processor = LangGraphFormProcessor(db)
        
        # 测试工作流和步骤ID
        test_workflow_id = "test-workflow-123"
        test_step_key = "applicant_setup"
        
        print("开始测试表单处理...")
        print(f"工作流ID: {test_workflow_id}")
        print(f"步骤键: {test_step_key}")
        print(f"HTML长度: {len(test_html)} 字符")
        print(f"用户资料键: {list(test_profile.keys())}")
        
        # 处理表单
        result = processor.process_form(
            workflow_id=test_workflow_id,
            step_key=test_step_key,
            form_html=test_html,
            profile_data=test_profile
        )
        
        print("\n=== 处理结果 ===")
        print(f"成功: {result['success']}")
        
        if result['success']:
            print(f"数据项数量: {len(result.get('data', []))}")
            print(f"动作数量: {len(result.get('actions', []))}")
            
            # 显示数据结构示例
            if result.get('data'):
                print("\n=== 数据结构示例 ===")
                first_item = result['data'][0]
                print(json.dumps(first_item, indent=2, ensure_ascii=False))
            
            # 显示动作示例
            if result.get('actions'):
                print("\n=== 动作示例 ===")
                first_action = result['actions'][0]
                print(json.dumps(first_action, indent=2, ensure_ascii=False))
        else:
            print(f"错误: {result.get('error', '未知错误')}")
        
        # 检查数据库中保存的数据
        print("\n=== 检查数据库数据 ===")
        step_repo = StepInstanceRepository(db)
        step = step_repo.get_step_by_key(test_workflow_id, test_step_key)
        
        if step and step.data:
            print("数据库中的数据结构:")
            saved_data = step.data
            
            # 检查预期的字段
            expected_fields = ["form_data", "actions", "questions", "metadata", "history"]
            for field in expected_fields:
                if field in saved_data:
                    if isinstance(saved_data[field], list):
                        print(f"✓ {field}: {len(saved_data[field])} 项")
                    elif isinstance(saved_data[field], dict):
                        print(f"✓ {field}: 字典，键: {list(saved_data[field].keys())}")
                    else:
                        print(f"✓ {field}: {type(saved_data[field])}")
                else:
                    print(f"✗ {field}: 缺失")
            
            # 显示完整的数据结构
            print("\n=== 完整数据结构 ===")
            print(json.dumps(saved_data, indent=2, ensure_ascii=False))
            
        else:
            print("数据库中没有找到步骤数据")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()

if __name__ == "__main__":
    test_data_structure() 