#!/usr/bin/env python3
"""
签证自动填表工作流演示脚本

演示如何使用LangGraph集成的HTML分析和AI自动填表功能
"""

import requests
import json
import time
from typing import Dict, Any

# API基础URL
BASE_URL = "http://localhost:8000"

# 示例HTML表单
SAMPLE_FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Visa Application Form</title>
</head>
<body>
    <form action="/submit" method="post">
        <h2>Personal Information</h2>
        
        <label for="firstName">First Name:</label>
        <input type="text" id="firstName" name="firstName" required>
        
        <label for="lastName">Last Name:</label>
        <input type="text" id="lastName" name="lastName" required>
        
        <label for="email">Email Address:</label>
        <input type="email" id="email" name="email" required>
        
        <label for="dateOfBirth">Date of Birth:</label>
        <input type="date" id="dateOfBirth" name="dateOfBirth" required>
        
        <label for="nationality">Nationality:</label>
        <select id="nationality" name="nationality" required>
            <option value="">Select Nationality</option>
            <option value="CN">Chinese</option>
            <option value="US">American</option>
            <option value="UK">British</option>
            <option value="IN">Indian</option>
        </select>
        
        <label for="passportNumber">Passport Number:</label>
        <input type="text" id="passportNumber" name="passportNumber" required>
        
        <label for="phone">Phone Number:</label>
        <input type="tel" id="phone" name="phone">
        
        <h3>Address Information</h3>
        
        <label for="address1">Address Line 1:</label>
        <input type="text" id="address1" name="address1" required>
        
        <label for="address2">Address Line 2:</label>
        <input type="text" id="address2" name="address2">
        
        <label for="city">City:</label>
        <input type="text" id="city" name="city" required>
        
        <label for="postalCode">Postal Code:</label>
        <input type="text" id="postalCode" name="postalCode" required>
        
        <h3>Application Type</h3>
        
        <input type="radio" id="tourist" name="visaType" value="tourist">
        <label for="tourist">Tourist Visa</label>
        
        <input type="radio" id="business" name="visaType" value="business">
        <label for="business">Business Visa</label>
        
        <input type="radio" id="student" name="visaType" value="student">
        <label for="student">Student Visa</label>
        
        <input type="submit" value="Submit Application">
    </form>
</body>
</html>
"""

# 示例用户档案数据
SAMPLE_USER_PROFILE = {
    "personalDetails": {
        "givenName": "张",
        "familyName": "三",
        "fullNameAsOnPassport": "ZHANG SAN",
        "dateOfBirth": "1990-05-15",
        "nationality": "CN",
        "placeOfBirth": "Beijing, China"
    },
    "contactInformation": {
        "emails": [
            {
                "emailAddress": "zhangsan@example.com",
                "isPrimary": True
            }
        ],
        "telephoneNumbers": [
            {
                "number": "+86-138-0013-8000",
                "type": "mobile"
            }
        ],
        "homeAddress": {
            "addressLine1": "123 Main Street",
            "addressLine2": "Apartment 4B",
            "city": "Beijing",
            "postalCode": "100000",
            "country": "China"
        }
    },
    "identityDocuments": {
        "passport": {
            "passportNumber": "E12345678",
            "issueDate": "2020-01-01",
            "expiryDate": "2030-01-01",
            "issuingCountry": "China"
        }
    }
}

class WorkflowDemo:
    """工作流演示类"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_user(self, username: str, email: str) -> str:
        """创建用户（简化版本 - 现在只需要生成用户ID字符串）"""
        print(f"🔄 生成用户标识: {username}")
        
        # 现在只需要生成一个用户ID字符串，不需要实际创建用户记录
        user_id = f"{username}_{int(time.time())}"
        print(f"✅ 用户ID生成成功: {user_id}")
        return user_id
    
    def create_workflow(self, user_id: str) -> str:
        """创建工作流实例"""
        print(f"🔄 为用户 {user_id} 创建工作流")
        
        payload = {
            "user_id": user_id,
            "workflow_definition_id": None,
            "initial_data": {}
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/workflows/",
                json=payload
            )
            response.raise_for_status()
            
            workflow_data = response.json()
            workflow_id = workflow_data["workflow_instance_id"]
            print(f"✅ 工作流创建成功，ID: {workflow_id}")
            return workflow_id
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 工作流创建失败: {e}")
            raise
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        print(f"🔄 获取工作流 {workflow_id} 状态")
        
        try:
            response = self.session.get(f"{self.base_url}/workflows/{workflow_id}/")
            response.raise_for_status()
            
            workflow_data = response.json()
            print(f"✅ 工作流状态: {workflow_data['status']}")
            print(f"   当前步骤: {workflow_data.get('current_step_key', 'N/A')}")
            print(f"   步骤数量: {len(workflow_data.get('steps', []))}")
            
            return workflow_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取工作流状态失败: {e}")
            raise
    
    def process_form_html(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理HTML表单"""
        print(f"🔄 处理步骤 {step_key} 的HTML表单")
        print(f"   HTML长度: {len(form_html)} 字符")
        
        if profile_data is None:
            profile_data = SAMPLE_USER_PROFILE
        
        try:
            response = self.session.post(
                f"{self.base_url}/workflows/{workflow_id}/steps/{step_key}/process-form",
                json={
                    "form_html": form_html,
                    "profile_data": profile_data
                }
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["success"]:
                print(f"✅ 表单处理成功")
                print(f"   生成操作: {len(result['actions'])} 个")
                print(f"   检测字段: {result.get('fields_detected', 0)} 个")
                print(f"   生成问题: {len(result.get('questions', []))} 个")
                print(f"   AI答案: {len(result.get('answers', []))} 个")
                
                # 显示生成的操作（Google插件格式）
                for i, action in enumerate(result["actions"][:3]):  # 只显示前3个
                    print(f"   操作 {i+1}: {action['action_type']} -> {action['selector']}")
                    if action.get('value'):
                        print(f"           值: {action['value']}")
                
                if len(result["actions"]) > 3:
                    print(f"   ... 还有 {len(result['actions']) - 3} 个操作")
                
                # 显示一些问题示例
                questions = result.get('questions', [])
                if questions:
                    print(f"\n   📝 问题示例:")
                    for i, question in enumerate(questions[:2]):  # 显示前2个
                        print(f"      Q{i+1}: {question.get('question', 'N/A')}")
                        print(f"          字段: {question.get('field_name', 'N/A')}")
                
                # 显示一些答案示例
                answers = result.get('answers', [])
                if answers:
                    print(f"\n   🤖 AI答案示例:")
                    for i, answer in enumerate(answers[:2]):  # 显示前2个
                        print(f"      A{i+1}: {answer.get('answer', 'N/A')}")
                        print(f"          置信度: {answer.get('confidence', 0)}%")
                        print(f"          字段: {answer.get('field_name', 'N/A')}")
                        
            else:
                print(f"❌ 表单处理失败: {result.get('error_details', 'Unknown error')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 表单处理请求失败: {e}")
            raise
    
    def get_step_analysis(self, workflow_id: str, step_key: str) -> Dict[str, Any]:
        """获取步骤分析数据"""
        print(f"🔄 获取步骤 {step_key} 的分析数据")
        
        try:
            response = self.session.get(
                f"{self.base_url}/workflows/{workflow_id}/steps/{step_key}/analysis"
            )
            response.raise_for_status()
            
            analysis_data = response.json()
            
            print(f"✅ 分析数据获取成功")
            print(f"   包含HTML分析: {analysis_data['has_html_analysis']}")
            print(f"   包含AI处理: {analysis_data['has_ai_processing']}")
            print(f"   包含表单操作: {analysis_data['has_form_actions']}")
            print(f"   字段数量: {analysis_data['field_count']}")
            print(f"   问题数量: {analysis_data['question_count']}")
            print(f"   答案数量: {analysis_data['answer_count']}")
            print(f"   操作数量: {analysis_data['action_count']}")
            
            return analysis_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取分析数据失败: {e}")
            raise
    
    def get_step_questions(self, workflow_id: str, step_key: str) -> list:
        """获取步骤问题"""
        print(f"🔄 获取步骤 {step_key} 的AI问题")
        
        try:
            response = self.session.get(
                f"{self.base_url}/workflows/{workflow_id}/steps/{step_key}/questions"
            )
            response.raise_for_status()
            
            questions = response.json()
            
            print(f"✅ 获取到 {len(questions)} 个问题")
            for i, question in enumerate(questions[:3]):  # 只显示前3个
                print(f"   问题 {i+1}: {question.get('question', 'N/A')}")
                print(f"           字段: {question.get('field_name', 'N/A')}")
                print(f"           必填: {question.get('required', False)}")
            
            if len(questions) > 3:
                print(f"   ... 还有 {len(questions) - 3} 个问题")
            
            return questions
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取问题失败: {e}")
            raise
    
    def get_step_answers(self, workflow_id: str, step_key: str) -> list:
        """获取步骤答案"""
        print(f"🔄 获取步骤 {step_key} 的AI答案")
        
        try:
            response = self.session.get(
                f"{self.base_url}/workflows/{workflow_id}/steps/{step_key}/answers"
            )
            response.raise_for_status()
            
            answers = response.json()
            
            print(f"✅ 获取到 {len(answers)} 个答案")
            for i, answer in enumerate(answers[:3]):  # 只显示前3个
                print(f"   答案 {i+1}: {answer.get('answer', 'N/A')}")
                print(f"           字段: {answer.get('field_name', 'N/A')}")
                print(f"           置信度: {answer.get('confidence', 0)}%")
                print(f"           推理: {answer.get('reasoning', 'N/A')[:50]}...")
            
            if len(answers) > 3:
                print(f"   ... 还有 {len(answers) - 3} 个答案")
            
            return answers
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 获取答案失败: {e}")
            raise
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 开始签证自动填表工作流演示")
        print("=" * 60)
        
        try:
            # 1. 创建用户
            user_id = self.create_user("demo_user", "demo@example.com")
            
            # 2. 创建工作流
            workflow_id = self.create_workflow(user_id)
            
            # 3. 获取工作流状态
            workflow_status = self.get_workflow_status(workflow_id)
            
            # 4. 处理HTML表单（使用personal_details步骤）
            step_key = "personal_details"
            form_result = self.process_form_html(workflow_id, step_key, SAMPLE_FORM_HTML)
            
            # 5. 获取分析数据
            analysis_data = self.get_step_analysis(workflow_id, step_key)
            
            # 6. 获取AI问题
            questions = self.get_step_questions(workflow_id, step_key)
            
            # 7. 获取AI答案
            answers = self.get_step_answers(workflow_id, step_key)
            
            print("\n" + "=" * 60)
            print("🎉 演示完成！")
            print(f"工作流ID: {workflow_id}")
            print(f"处理步骤: {step_key}")
            print(f"生成操作数: {len(form_result.get('actions', []))}")
            print(f"AI问题数: {len(questions)}")
            print(f"AI答案数: {len(answers)}")
            print(f"检测字段数: {form_result.get('fields_detected', 0)}")
            
            # 显示处理元数据
            metadata = form_result.get('processing_metadata', {})
            if metadata.get('processed_at'):
                print(f"处理时间: {metadata['processed_at']}")
            
            return {
                "workflow_id": workflow_id,
                "step_key": step_key,
                "form_result": form_result,
                "analysis_data": analysis_data,
                "questions": questions,
                "answers": answers
            }
            
        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {e}")
            raise

def main():
    """主函数"""
    print("签证自动填表工作流演示")
    print("请确保API服务器正在运行在 http://localhost:8000")
    
    # 检查服务器是否可用
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API服务器连接正常")
        else:
            print("❌ API服务器响应异常")
            return
    except requests.exceptions.RequestException:
        print("❌ 无法连接到API服务器，请确保服务器正在运行")
        return
    
    # 运行演示
    demo = WorkflowDemo()
    try:
        result = demo.run_complete_demo()
        
        # 保存结果到文件
        with open("demo_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📄 演示结果已保存到 demo_result.json")
        
    except Exception as e:
        print(f"演示失败: {e}")

if __name__ == "__main__":
    main() 