#!/usr/bin/env python3
"""
创建层级化工作流定义
根据用户提供的层级化步骤结构创建新的工作流定义
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.database.database_config import get_db_session
from src.database.workflow_repositories import WorkflowDefinitionRepository

# 层级化工作流定义
HIERARCHICAL_WORKFLOW_DEFINITION = [
    {
        "key": "identity_and_contact",
        "name": "1. Identity and contact",
        "order": 1,
        "sub_steps": [
            {
                "key": "confirm_identity",
                "name": "Confirm your identity",
                "order": 1
            },
            {
                "key": "linking_code",
                "name": "Linking code for family members",
                "order": 2
            },
            {
                "key": "immigration_adviser",
                "name": "Immigration adviser details",
                "order": 3
            },
            {
                "key": "contact_preferences",
                "name": "Contact preferences",
                "order": 4
            },
            {
                "key": "other_names",
                "name": "Other names and nationalities",
                "order": 5
            }
        ]
    },
    {
        "key": "prepare_application",
        "name": "2. Prepare application",
        "order": 2,
        "sub_steps": [
            {
                "key": "people_applying",
                "name": "People applying with you",
                "order": 1
            },
            {
                "key": "your_location",
                "name": "Your location",
                "order": 2
            },
            {
                "key": "work_details",
                "name": "Work details",
                "order": 3
            },
            {
                "key": "personal_details",
                "name": "Personal details",
                "order": 4
            },
            {
                "key": "family_relationships",
                "name": "Family and relationships",
                "order": 5
            },
            {
                "key": "travel_history",
                "name": "Travel history",
                "order": 6
            },
            {
                "key": "criminality",
                "name": "Criminality",
                "order": 7
            },
            {
                "key": "financial_maintenance",
                "name": "Financial maintenance",
                "order": 8
            },
            {
                "key": "english_language",
                "name": "English language ability",
                "order": 9
            },
            {
                "key": "security_questions",
                "name": "Account security questions",
                "order": 10
            },
            {
                "key": "declaration",
                "name": "Declaration",
                "order": 11
            }
        ]
    },
    {
        "key": "pay_and_submit",
        "name": "3. Pay and submit application",
        "order": 3,
        "sub_steps": [
            {
                "key": "immigration_health_surcharge",
                "name": "Immigration health surcharge",
                "order": 1
            },
            {
                "key": "application_fee",
                "name": "Application fee",
                "order": 2
            },
            {
                "key": "submit_application",
                "name": "Submit application",
                "order": 3
            }
        ]
    }
]

def create_hierarchical_workflow():
    """创建层级化工作流定义"""
    
    print("开始创建层级化工作流定义...")
    
    # 获取数据库会话
    from src.database.database_config import db_config
    
    with db_config.get_session() as db_session:
        try:
            # 创建工作流定义仓库
            definition_repo = WorkflowDefinitionRepository(db_session)
            
            # 创建工作流定义
            workflow_def = definition_repo.create_definition(
                name="UK Visa Application - Hierarchical Workflow",
                description="英国签证申请层级化工作流，包含3个主步骤和16个子步骤",
                step_definitions=HIERARCHICAL_WORKFLOW_DEFINITION
            )
            
            print(f"✅ 成功创建层级化工作流定义!")
            print(f"   工作流定义ID: {workflow_def.workflow_definition_id}")
            print(f"   工作流名称: {workflow_def.name}")
            print(f"   创建时间: {workflow_def.created_at}")
            
            # 统计步骤数量
            main_steps_count = len(HIERARCHICAL_WORKFLOW_DEFINITION)
            sub_steps_count = sum(len(step.get('sub_steps', [])) for step in HIERARCHICAL_WORKFLOW_DEFINITION)
            
            print(f"   主步骤数量: {main_steps_count}")
            print(f"   子步骤数量: {sub_steps_count}")
            print(f"   总步骤数量: {main_steps_count + sub_steps_count}")
            
            # 显示步骤结构
            print("\n📋 工作流步骤结构:")
            for main_step in HIERARCHICAL_WORKFLOW_DEFINITION:
                print(f"   {main_step['order']}. {main_step['name']} ({main_step['key']})")
                for sub_step in main_step.get('sub_steps', []):
                    print(f"      {sub_step['order']}. {sub_step['name']} ({sub_step['key']})")
            
            print(f"\n🎉 层级化工作流定义创建完成!")
            print(f"   可以使用工作流定义ID '{workflow_def.workflow_definition_id}' 来创建工作流实例")
            
            return workflow_def.workflow_definition_id
            
        except Exception as e:
            print(f"❌ 创建工作流定义失败: {str(e)}")
            raise e

def test_workflow_creation(workflow_definition_id: str):
    """测试使用新的工作流定义创建工作流实例"""
    
    print(f"\n开始测试工作流实例创建...")
    
    # 获取数据库会话
    from src.database.database_config import db_config
    
    with db_config.get_session() as db_session:
        try:
            from src.business.workflow_service import WorkflowService
            from src.model.workflow_schemas import WorkflowInitiationPayload
            
            # 创建工作流服务
            workflow_service = WorkflowService(db_session)
            
            # 创建工作流实例
            payload = WorkflowInitiationPayload(
                user_id="test_user_hierarchical",
                workflow_definition_id=workflow_definition_id,
                initial_data={"test": "hierarchical_workflow"}
            )
            
            workflow_instance = workflow_service.create_workflow(payload)
            
            print(f"✅ 成功创建工作流实例!")
            print(f"   工作流实例ID: {workflow_instance.workflow_instance_id}")
            print(f"   用户ID: {workflow_instance.user_id}")
            print(f"   状态: {workflow_instance.status}")
            print(f"   当前步骤: {workflow_instance.current_step_key}")
            
            # 获取工作流详细状态
            workflow_detail = workflow_service.get_workflow_status(workflow_instance.workflow_instance_id)
            
            print(f"\n📊 工作流详细信息:")
            print(f"   总步骤数: {len(workflow_detail.steps)}")
            
            # 显示步骤层级结构
            print(f"\n📋 步骤实例结构:")
            for step in workflow_detail.steps:
                print(f"   {step.order}. {step.name} ({step.step_key}) - 状态: {step.status.value} - 类型: {step.step_type.value}")
                for child_step in step.child_steps:
                    print(f"      {child_step.order}. {child_step.name} ({child_step.step_key}) - 状态: {child_step.status.value} - 类型: {child_step.step_type.value}")
            
            print(f"\n🎉 层级化工作流实例测试完成!")
            
            return workflow_instance.workflow_instance_id
            
        except Exception as e:
            print(f"❌ 测试工作流实例创建失败: {str(e)}")
            raise e

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="创建层级化工作流定义")
    parser.add_argument("--test", action="store_true", help="创建后立即测试工作流实例")
    
    args = parser.parse_args()
    
    try:
        # 创建工作流定义
        workflow_definition_id = create_hierarchical_workflow()
        
        # 如果指定了测试参数，则创建测试实例
        if args.test:
            test_workflow_creation(workflow_definition_id)
            
    except Exception as e:
        print(f"❌ 操作失败: {str(e)}")
        sys.exit(1) 