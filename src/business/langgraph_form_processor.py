import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, TypedDict, Tuple

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import SecretStr, BaseModel
from sqlalchemy.orm import Session

from src.database.workflow_repositories import (
    StepInstanceRepository
)
from src.model.workflow_entities import StepStatus, WorkflowInstance


def clean_llm_response(response_content: str) -> str:
    """Clean and extract JSON from LLM response"""
    content = response_content.strip()

    # Remove markdown code blocks
    if content.startswith('```json'):
        content = content[7:]
    elif content.startswith('```'):
        content = content[3:]

    if content.endswith('```'):
        content = content[:-3]

    content = content.strip()

    # Try to find JSON object in the content
    # Look for { ... } pattern
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        content = json_match.group(0)

    # Clean up common formatting issues
    content = re.sub(r'\n\s*', ' ', content)  # Remove newlines and extra spaces
    content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
    content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays

    return content


def robust_json_parse(response_content: str) -> dict:
    """Robustly parse JSON from LLM response with multiple fallback strategies"""

    # Strategy 1: Direct parsing
    try:
        return json.loads(response_content)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Clean and parse
    try:
        cleaned_content = clean_llm_response(response_content)
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Extract JSON patterns manually
    try:
        # Look for answer, confidence, reasoning, needs_intervention pattern
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', response_content)
        confidence_match = re.search(r'"confidence"\s*:\s*(\d+)', response_content)
        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response_content)
        intervention_match = re.search(r'"needs_intervention"\s*:\s*(true|false)', response_content)

        if answer_match and confidence_match:
            return {
                "answer": answer_match.group(1),
                "confidence": int(confidence_match.group(1)),
                "reasoning": reasoning_match.group(1) if reasoning_match else "Parsed from partial response",
                "needs_intervention": intervention_match.group(1).lower() == 'true' if intervention_match else False
            }
    except Exception:
        pass

    # Strategy 4: Look for actions array pattern
    try:
        actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', response_content, re.DOTALL)
        if actions_match:
            actions_content = actions_match.group(1)
            # Try to parse individual actions
            action_objects = []
            action_pattern = r'\{[^}]*\}'
            for action_match in re.finditer(action_pattern, actions_content):
                try:
                    action_obj = json.loads(action_match.group(0))
                    action_objects.append(action_obj)
                except:
                    pass

            if action_objects:
                return {"actions": action_objects}
    except Exception:
        pass

    # If all strategies fail, return a default structure
    return {
        "answer": "",
        "confidence": 0,
        "reasoning": f"Failed to parse LLM response: {response_content[:100]}...",
        "needs_intervention": True
    }


class FormAnalysisState(TypedDict):
    """State for form analysis workflow"""
    workflow_id: str
    step_key: str
    form_html: str
    profile_data: Dict[str, Any]
    profile_dummy_data: Dict[str, Any]  # Add dummy data field
    parsed_form: Optional[Dict[str, Any]]
    detected_fields: List[Dict[str, Any]]
    field_questions: List[Dict[str, Any]]
    ai_answers: List[Dict[str, Any]]
    merged_qa_data: List[Dict[str, Any]]  # New field for merged question-answer data
    form_actions: List[Dict[str, Any]]
    llm_generated_actions: List[Dict[str, Any]]  # New field for LLM-generated actions
    saved_step_data: Optional[Dict[str, Any]]  # New field to store data saved to database
    dummy_data_usage: List[Dict[str, Any]]  # Track which questions used dummy data
    analysis_complete: bool
    error_details: Optional[str]  # Changed from error_message to error_details
    messages: List[Dict[str, Any]]


class FormFieldQuestion(BaseModel):
    """Form field question model"""
    field_selector: str
    field_name: str
    field_type: str
    field_label: str
    question: str
    required: bool
    options: Optional[List[Dict[str, str]]] = None


class AIAnswer(BaseModel):
    """AI generated answer model"""
    question_id: str
    field_selector: str
    answer: str
    confidence: float
    reasoning: str


class FormAction(BaseModel):
    """Form action model"""
    selector: str
    action_type: str  # input, click
    value: Optional[str] = None
    order: int


class StepAnalyzer:
    """步骤分析器，用于分析当前页面属于哪个步骤"""
    
    def __init__(self, db_session: Session):
        """Initialize analyzer with database session"""
        self.db = db_session
        self.step_repo = StepInstanceRepository(db_session)  # Add step repository
        self.llm = ChatOpenAI(
            base_url=os.getenv("MODEL_BASE_URL"),
            api_key=SecretStr(os.getenv("MODEL_API_KEY")),
            model=os.getenv("MODEL_WORKFLOW_NAME"),
            temperature=0,
        )
        self._page_context = {}  # Store page context for analysis
        self._current_workflow_id = None  # Thread isolation for LLM calls

    def set_workflow_id(self, workflow_id: str):
        """Set the current workflow ID for thread isolation"""
        self._current_workflow_id = workflow_id

    def _invoke_llm(self, messages: List, workflow_id: str = None):
        """Invoke LLM (workflow_id kept for logging purposes only)"""
        # workflow_id is kept for logging but not used in LLM call
        used_workflow_id = workflow_id or self._current_workflow_id
        if used_workflow_id:
            print(f"[workflow_id:{used_workflow_id}] DEBUG: Invoking LLM")
        return self.llm.invoke(messages)
    
    def analyze_step(self, html_content: str, workflow_id: str, current_step_key: str) -> Dict[str, Any]:
        """
        分析当前页面属于当前步骤还是下一步骤
        
        改进逻辑：
        - 同时获取当前步骤和下一步骤的上下文信息
        - 使用 LLM 比较页面内容更符合哪个步骤
        - 如果属于下一步，完成当前步骤并激活下一步
        """
        try:
            # 提取页面问题
            page_analysis = self._extract_page_questions(html_content)
            
            # 获取当前步骤上下文
            current_step_context = self._get_step_context(workflow_id, current_step_key)
            
            # 获取下一步骤上下文
            next_step_key = self._find_next_step(workflow_id, current_step_key)
            next_step_context = None
            if next_step_key:
                next_step_context = self._get_step_context(workflow_id, next_step_key)
            
            # 使用 LLM 进行比较分析
            analysis_result = self._analyze_with_llm(
                page_analysis=page_analysis,
                current_step_context=current_step_context,
                next_step_context=next_step_context,
                current_step_key=current_step_key,
                next_step_key=next_step_key
            )
            
            # 如果页面属于下一步骤，执行步骤转换
            if analysis_result.get("belongs_to_next_step", False) and next_step_key:
                print(
                    f"[workflow_id:{workflow_id}] DEBUG: Page belongs to next step {next_step_key}, executing step transition")
                
                # 获取当前步骤实例
                current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
                if current_step:
                    # 1. 完成当前步骤
                    self.step_repo.update_step_status(current_step.step_instance_id, StepStatus.COMPLETED_SUCCESS)
                    current_step.completed_at = datetime.utcnow()
                    print(f"[workflow_id:{workflow_id}] DEBUG: Completed current step {current_step_key}")
                    
                    # 2. 激活下一个步骤
                    next_step = self.step_repo.get_step_by_key(workflow_id, next_step_key)
                    if next_step:
                        self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
                        next_step.started_at = datetime.utcnow()
                        print(f"[workflow_id:{workflow_id}] DEBUG: Activated next step {next_step_key}")
                        
                        # 3. 更新工作流实例的当前步骤
                        from src.database.workflow_repositories import WorkflowInstanceRepository
                        instance_repo = WorkflowInstanceRepository(self.db)
                        instance_repo.update_instance_status(
                            workflow_id, 
                            None,  # 保持当前工作流状态
                            next_step_key  # 更新当前步骤键
                        )
                        print(f"[workflow_id:{workflow_id}] DEBUG: Updated workflow current step to {next_step_key}")
                        
                        # 4. 更新分析结果，指示应该使用下一步骤执行
                        analysis_result.update({
                            "should_use_next_step": True,
                            "next_step_key": next_step_key,
                            "next_step_instance_id": next_step.step_instance_id,
                            "step_transition_completed": True
                        })
                
                # 提交数据库更改
                self.db.commit()
                print(f"[workflow_id:{workflow_id}] DEBUG: Step transition completed and committed")
            else:
                # 页面属于当前步骤，继续使用当前步骤
                analysis_result.update({
                    "should_use_next_step": False,
                    "step_transition_completed": False
                })
            
            return analysis_result
            
        except Exception as e:
            print(f"Error analyzing step: {str(e)}")
            # 发生错误时回滚数据库更改
            self.db.rollback()
            return {
                "belongs_to_current_step": True,
                "belongs_to_next_step": False,
                "should_use_next_step": False,
                "main_question": "",
                "step_transition_completed": False,
                "next_step_key": None,
                "reasoning": f"Error analyzing step: {str(e)}"
            }
    
    def _extract_page_questions(self, html_content: str) -> Dict[str, Any]:
        """提取页面问题和上下文信息"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        analysis = {
            "page_title": "",
            "form_title": "",
            "main_heading": "",
            "questions": [],  # 页面上的所有问题
            "form_elements": []
        }
        
        # 提取页面标题
        title_tag = soup.find("title")
        if title_tag:
            analysis["page_title"] = title_tag.get_text(strip=True)
        
        # 提取主标题
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                analysis["main_heading"] = heading.get_text(strip=True)
                break
        
        # 提取表单标题
        form = soup.find("form")
        if form:
            legend = form.find("legend")
            if legend:
                analysis["form_title"] = legend.get_text(strip=True)
            
            # 提取表单元素和对应的问题
            for element in form.find_all(["input", "select", "textarea"]):
                # 获取字段标签
                label = self._find_field_label(element)
                
                # 构建问题
                question = {
                    "field_name": element.get("name", ""),
                    "field_type": element.get("type", "text"),
                    "question_text": label or element.get("placeholder", ""),
                    "required": element.get("required") is not None,
                    "options": self._extract_field_options(element)
                }
                
                analysis["questions"].append(question)
                
                # 保存表单元素信息
                element_info = {
                    "type": element.name,
                    "name": element.get("name", ""),
                    "id": element.get("id", ""),
                    "label": label,
                    "required": element.get("required") is not None
                }
                analysis["form_elements"].append(element_info)
        
        return analysis
    
    def _find_field_label(self, element) -> str:
        """Find label text for form field using multiple strategies"""
        # Strategy 1: Try aria-label first (most explicit)
        aria_label = element.get("aria-label", "").strip()
        if aria_label:
            return aria_label

        # Strategy 2: For radio/checkbox fields, look for specific option label, NOT fieldset legend
        field_type = element.get("type", "").lower()
        if field_type in ["radio", "checkbox"]:
            # For radio/checkbox, we want the specific option text, not the general question
            # Skip fieldset legend and look for the specific option's label
            pass  # We'll use other strategies to find the specific option text

        # Strategy 3: Try to find associated label by ID
        field_id = element.get("id", "").strip()
        if field_id:
            # Look for label with for attribute
            label_element = element.find_parent().find("label", {"for": field_id}) if element.find_parent() else None
            if not label_element:
                # Search in the whole document
                soup_root = element
                while soup_root.parent:
                    soup_root = soup_root.parent
                label_element = soup_root.find("label", {"for": field_id})

            if label_element:
                label_text = label_element.get_text(strip=True)
                if label_text:
                    return label_text

        # Strategy 4: Look for label that contains this element
        current = element
        for _ in range(3):  # Check up to 3 levels up
            parent = current.find_parent()
            if not parent:
                break
            if parent.name == "label":
                label_text = parent.get_text(strip=True)
                # Remove the element's own text to get clean label
                element_text = element.get_text(strip=True) if hasattr(element, 'get_text') else ""
                if element_text and element_text in label_text:
                    label_text = label_text.replace(element_text, "").strip()
                if label_text:
                    return label_text
            current = parent

        # Strategy 5: Look for nearby label (preceding or following)
        # Check previous siblings
        prev_element = element.find_previous_sibling("label")
        if prev_element:
            label_text = prev_element.get_text(strip=True)
            if label_text:
                return label_text

        # Check next siblings (for cases where label comes after)
        next_element = element.find_next_sibling("label")
        if next_element:
            label_text = next_element.get_text(strip=True)
            if label_text:
                return label_text

        # Strategy 6: Look for nearby text content (enhanced for radio/checkbox)
        parent = element.find_parent()
        if parent:
            # For radio/checkbox, look for associated text more aggressively
            if field_type in ["radio", "checkbox"]:
                # Look for next sibling span/div with text (common pattern)
                next_sibling = element.find_next_sibling()
                while next_sibling:
                    if hasattr(next_sibling, 'name') and next_sibling.name in ['span', 'div', 'p', 'label']:
                        text = next_sibling.get_text(strip=True)
                        if text and len(text) < 200:  # Reasonable label length for radio options
                            print(
                                f"DEBUG: _find_field_label - Found option text for {element.get('value', 'unknown')}: '{text}'")
                            return text
                    next_sibling = next_sibling.find_next_sibling()

                # Look for text within the same parent container
                for child in parent.children:
                    if hasattr(child, 'name') and child.name in ['span', 'div', 'p']:
                        text = child.get_text(strip=True)
                        if text and len(text) < 200 and text not in ["(Required)", "Required"]:
                            print(
                                f"DEBUG: _find_field_label - Found option text in parent for {element.get('value', 'unknown')}: '{text}'")
                            return text
            else:
                # Original logic for other field types
                for child in parent.children:
                    if hasattr(child, 'name') and child.name in ['span', 'div', 'p']:
                        text = child.get_text(strip=True)
                        if text and len(text) < 100:  # Reasonable label length
                            return text

        # Strategy 7: Use placeholder attribute
        placeholder = element.get("placeholder", "").strip()
        if placeholder:
            return placeholder

        # Strategy 8: Use title attribute
        title = element.get("title", "").strip()
        if title:
            return title

        # Strategy 9: Generate from name attribute
        name = element.get("name", "").strip()
        if name:
            # Convert name to readable format
            readable_name = name.replace("_", " ").replace("-", " ").replace(".", " ")
            readable_name = " ".join(word.capitalize() for word in readable_name.split())
            return readable_name

        # Strategy 10: Generate from id attribute
        if field_id:
            readable_id = field_id.replace("_", " ").replace("-", " ").replace(".", " ")
            readable_id = " ".join(word.capitalize() for word in readable_id.split())
            return readable_id

        # Last resort: return element type
        element_type = element.get("type", element.name) if element.name == "input" else element.name
        return f"{element_type.capitalize()} Field"
    
    def _extract_field_options(self, element) -> List[Dict[str, str]]:
        """提取字段的选项"""
        options = []
        
        if element.name == "select":
            for option in element.find_all("option"):
                option_value = option.get("value", "")
                option_text = option.get_text(strip=True)
                if option_value or option_text:
                    options.append({
                        "value": option_value or option_text,
                        "text": option_text or option_value
                    })
        elif element.get("type") in ["radio", "checkbox"]:
            # 对于单选和复选框，查找相关的选项
            name = element.get("name", "")
            if name:
                parent = element.find_parent()
                if parent:
                    related_elements = parent.find_all(
                        "input", 
                        {"type": element.get("type"), "name": name}
                    )
                    for related in related_elements:
                        value = related.get("value", "")
                        label = self._find_field_label(related)
                        if value or label:
                            options.append({
                                "value": value or label,
                                "text": label or value
                            })
        
        return options
    
    def _get_step_context(self, workflow_id: str, step_key: str) -> Dict[str, Any]:
        """获取步骤上下文信息"""
        try:
            step = self.step_repo.get_step_by_key(workflow_id, step_key)
            if step:
                return {
                    "name": step.name,
                    "order": step.order,
                    "expected_questions": []  # 可以从步骤定义中获取
                }
            return {}
        except Exception as e:
            print(f"Error getting step context: {str(e)}")
            return {}

    def _find_next_step(self, workflow_id: str, current_step_key: str) -> Optional[str]:
        """根据步骤顺序找到下一个步骤"""
        try:
            # 获取所有步骤，按顺序排序
            all_steps = self.step_repo.get_workflow_steps(workflow_id)
            if not all_steps:
                return None
            
            # 找到当前步骤
            current_step = None
            for step in all_steps:
                if step.step_key == current_step_key:
                    current_step = step
                    break
            
            if not current_step:
                return None
            
            # 按顺序排序所有步骤
            sorted_steps = sorted(all_steps, key=lambda s: s.order or 0)
            
            # 找到下一个步骤
            current_order = current_step.order or 0
            for step in sorted_steps:
                if (step.order or 0) > current_order:
                    return step.step_key
            
            # 没有找到下一个步骤
            return None
            
        except Exception as e:
            print(f"Error finding next step: {str(e)}")
            return None
    
    def _analyze_with_llm(self, page_analysis: Dict[str, Any], 
                         current_step_context: Dict[str, Any],
                         next_step_context: Optional[Dict[str, Any]],
                         current_step_key: str,
                         next_step_key: Optional[str]) -> Dict[str, Any]:
        """使用 LLM 比较分析页面内容属于当前步骤还是下一步骤"""
        
        # 构建比较分析的提示词
        if next_step_context:
            prompt = f"""
            # Role
            You are a workflow step analyzer that determines which step a page belongs to by comparing it against multiple steps.

            # Current Step Context
            Step Key: {current_step_key}
            Step Name: {current_step_context.get('name', '')}
            Step Order: {current_step_context.get('order', '')}
            Expected Questions: {current_step_context.get('expected_questions', [])}

            # Next Step Context  
            Step Key: {next_step_key}
            Step Name: {next_step_context.get('name', '')}
            Step Order: {next_step_context.get('order', '')}
            Expected Questions: {next_step_context.get('expected_questions', [])}

            # Page Analysis
            Page Title: {page_analysis.get('page_title', '')}
            Form Title: {page_analysis.get('form_title', '')}
            Main Heading: {page_analysis.get('main_heading', '')}
            Questions: {json.dumps(page_analysis.get('questions', []), indent=2)}

            # Task
            Compare the page content against BOTH the current step and next step to determine which one it belongs to.

            # Analysis Rules:
            1. Analyze the semantic similarity between page content and each step's expected questions/topics
            2. Consider step names and their logical progression
            3. If page content clearly matches the NEXT step better than current step, mark belongs_to_next_step: true
            4. If page content matches the CURRENT step better or equally, mark belongs_to_current_step: true
            5. Use confidence scores to indicate how certain you are about the classification

            # Examples:
            - Current: "Personal Details", Next: "Contact Info", Page: "Enter phone number" → Next step
            - Current: "Personal Details", Next: "Contact Info", Page: "Enter first name" → Current step  
            - Current: "Education", Next: "Work Experience", Page: "Previous job title" → Next step
            - Current: "Education", Next: "Work Experience", Page: "University name" → Current step

            # Response Format (JSON)
            {{
                "belongs_to_current_step": true/false,
                "belongs_to_next_step": true/false,
                "current_step_confidence": 0.0-1.0,
                "next_step_confidence": 0.0-1.0,
                "main_question": "main question being asked on this page",
                "reasoning": "detailed comparison explaining why page belongs to current or next step",
                "recommended_action": "continue_current_step" or "transition_to_next_step"
            }}
            """
        else:
            # 如果没有下一步骤，使用原来的单步骤分析逻辑
            prompt = f"""
            # Role
            You are a workflow analyzer that determines if the current page belongs to a specific step in a workflow.

            # Current Step Context
            Step Key: {current_step_key}
            Step Name: {current_step_context.get('name', '')}
            Step Order: {current_step_context.get('order', '')}
            Expected Questions: {current_step_context.get('expected_questions', [])}

            # Page Analysis
            Page Title: {page_analysis.get('page_title', '')}
            Form Title: {page_analysis.get('form_title', '')}
            Main Heading: {page_analysis.get('main_heading', '')}
            Questions: {json.dumps(page_analysis.get('questions', []), indent=2)}

            # Task
            Analyze if the current page belongs to the current workflow step (no next step available).

            # Response Format (JSON)
            {{
                "belongs_to_current_step": true,
                "belongs_to_next_step": false,
                "current_step_confidence": 0.0-1.0,
                "next_step_confidence": 0.0,
                "main_question": "main question being asked on this page",
                "reasoning": "explanation of why this page belongs to the current step",
                "recommended_action": "continue_current_step"
            }}
            """

        response = self._invoke_llm([HumanMessage(content=prompt)])
        try:
            result = json.loads(response.content)
            
            # 验证和标准化结果
            if not isinstance(result.get("belongs_to_current_step"), bool):
                result["belongs_to_current_step"] = True
            if not isinstance(result.get("belongs_to_next_step"), bool):
                result["belongs_to_next_step"] = False
                
            # 确保逻辑一致性：如果属于下一步，就不属于当前步
            if result.get("belongs_to_next_step", False):
                result["belongs_to_current_step"] = False
                
            return result
            
        except json.JSONDecodeError:
            return {
                "belongs_to_current_step": True,
                "belongs_to_next_step": False,
                "current_step_confidence": 0.5,
                "next_step_confidence": 0.0,
                "main_question": page_analysis["questions"][0]["question_text"] if page_analysis["questions"] else "",
                "reasoning": "Failed to parse LLM response, assuming current step continues",
                "recommended_action": "continue_current_step"
            }
    
    def _extract_field_info(self, element) -> Optional[Dict[str, Any]]:
        """Extract information from HTML form element"""
        try:
            field_type = element.name.lower()

            # Handle input types more specifically
            if element.name == "input":
                field_type = element.get("type", "text").lower()

            print(f"DEBUG: _extract_field_info - Processing {element.name} element, type: {field_type}")
            print(f"DEBUG: _extract_field_info - Element attributes: name={element.get('name', '')}, id={element.get('id', '')}, class={element.get('class', '')}")

            # Skip CSRF token fields and other system fields
            field_name = element.get("name", "").lower()
            field_id = element.get("id", "").lower()
            
            # List of field names/patterns to ignore
            ignore_patterns = [
                "csrf", "csrftoken", "_token", "authenticity_token",
                "_method", "__viewstate", "__eventvalidation",
                "submit", "reset", "button"
            ]
            
            # Check if field should be ignored
            for pattern in ignore_patterns:
                if (pattern in field_name or 
                    pattern in field_id or 
                    field_type == pattern):
                    print(f"DEBUG: _extract_field_info - Ignoring system field: {field_name} (pattern: {pattern})")
                    return None
            
            # Also ignore hidden fields that look like system fields
            if field_type == "hidden":
                # Allow some hidden fields that might be user data
                allowed_hidden_patterns = ["user", "profile", "data", "info"]
                is_allowed = any(pattern in field_name for pattern in allowed_hidden_patterns)
                
                if not is_allowed:
                    print(f"DEBUG: _extract_field_info - Ignoring hidden system field: {field_name}")
                    return None

            field_info = {
                "name": element.get("name", ""),
                "type": field_type,
                "selector": self._generate_selector(element),
                "required": element.get("required") is not None,
                "label": self._find_field_label(element),
                "placeholder": element.get("placeholder", ""),
                "value": element.get("value", ""),
                "options": []
            }
            
            print(f"DEBUG: _extract_field_info - Basic field info: {field_info}")

            # Handle different field types and their options
            if field_type == "select":
                options = []
                for option in element.find_all("option"):
                    option_value = option.get("value", "")
                    option_text = option.get_text(strip=True)
                    if option_value or option_text:  # Include options even without explicit value
                        options.append({
                            "value": option_value or option_text,
                            "text": option_text or option_value
                        })
                field_info["options"] = options
                print(f"DEBUG: _extract_field_info - Select options: {options}")

            elif field_type == "radio":
                # For radio buttons, find all related radio buttons with same name
                radio_name = element.get("name", "")
                if radio_name:
                    # Find the container and look for related radio buttons
                    # Search in a wider scope to find all radio buttons with same name
                    container = element.find_parent()
                    search_scope = container
                    
                    # If container is too small, search in document root
                    while search_scope and len(search_scope.find_all('input', {'type': 'radio', 'name': radio_name})) < 2:
                        search_scope = search_scope.find_parent()
                        if not search_scope:
                            # Search in the entire document
                            search_scope = element
                            while search_scope.parent:
                                search_scope = search_scope.parent
                            break
                    
                    related_radios = search_scope.find_all('input', {'type': 'radio', 'name': radio_name}) if search_scope else []
                    options = []
                    for radio in related_radios:
                        label_text = self._find_field_label(radio)
                        radio_value = radio.get("value", "")
                        if radio_value or label_text:  # Only include if has value or label
                            options.append({
                                "value": radio_value or label_text,
                                "text": label_text or radio_value
                            })
                    
                    field_info["options"] = options
                    print(f"DEBUG: _extract_field_info - Radio options for '{radio_name}': {options}")

            elif field_type == "checkbox":
                # For checkboxes, find related checkboxes with same name
                checkbox_name = element.get("name", "")
                if checkbox_name:
                    # Find the container and look for related checkboxes
                    container = element.find_parent() or element
                    related_checkboxes = container.find_all('input', {'type': 'checkbox', 'name': checkbox_name})
                    options = []
                    for cb in related_checkboxes:
                        label_text = self._find_field_label(cb)
                        cb_value = cb.get("value", "")
                        options.append({
                            "value": cb_value or label_text,
                            "text": label_text or cb_value
                        })
                    field_info["options"] = options
                    print(f"DEBUG: _extract_field_info - Checkbox options: {options}")

            # Return fields with name, label, or meaningful selector - lowered threshold
            has_name = bool(field_info["name"])
            has_label = bool(field_info["label"])
            has_selector = bool(field_info["selector"])
            
            print(f"DEBUG: _extract_field_info - Validation: has_name={has_name}, has_label={has_label}, has_selector={has_selector}")
            
            if has_name or has_label or has_selector:
                print(f"DEBUG: _extract_field_info - Field accepted: {field_info}")
                return field_info
            else:
                print(f"DEBUG: _extract_field_info - Field rejected: no name, label, or selector")
                return None

        except Exception as e:
            print(f"DEBUG: _extract_field_info - Exception: {str(e)}")
            return None
    
    def _generate_selector(self, element) -> str:
        """Generate unique CSS selector for element"""
        # Strategy 1: Use ID selector (most unique)
        element_id = element.get("id", "").strip()
        if element_id:
            return f"#{element_id}"

        # Strategy 2: Use name attribute with element type for uniqueness
        element_name = element.get("name", "").strip()
        if element_name:
            element_type = element.name.lower()
            if element_type == "input":
                input_type = element.get("type", "text").lower()
                return f"input[type='{input_type}'][name='{element_name}']"
            else:
                return f"{element_type}[name='{element_name}']"

        # Strategy 3: Use combination of attributes for uniqueness
        element_type = element.name.lower()
        selectors = [element_type]

        # Add type for input elements
        if element_type == "input":
            input_type = element.get("type", "text").lower()
            selectors.append(f"[type='{input_type}']")

        # Add class if available
        class_attr = element.get("class")
        if class_attr:
            if isinstance(class_attr, list):
                # Use the first class for selector
                first_class = class_attr[0].strip()
                if first_class:
                    selectors.append(f".{first_class}")
            else:
                class_name = class_attr.strip()
                if class_name:
                    selectors.append(f".{class_name}")

        # Add placeholder as attribute selector if unique enough
        placeholder = element.get("placeholder", "").strip()
        if placeholder and len(placeholder) > 5:  # Only use meaningful placeholders
            selectors.append(f"[placeholder='{placeholder}']")

        # Add value for radio/checkbox to make it specific
        if element_type == "input":
            input_type = element.get("type", "").lower()
            if input_type in ["radio", "checkbox"]:
                value = element.get("value", "").strip()
                if value:
                    selectors.append(f"[value='{value}']")

        selector = "".join(selectors)

        # If we still don't have a good selector, try positional approach
        if selector == element_type:
            # Find position among siblings of same type
            parent = element.find_parent()
            if parent:
                siblings = parent.find_all(element_type, recursive=False)
                if len(siblings) > 1:
                    try:
                        index = siblings.index(element)
                        selector = f"{element_type}:nth-of-type({index + 1})"
                    except ValueError:
                        pass

        return selector or element_type
    
    def _extract_page_context(self, parsed_form: Dict[str, Any]) -> Dict[str, str]:
        """Extract page context information like titles, headings, descriptions"""
        context = {
            "page_title": "",
            "form_title": "",
            "main_heading": "",
            "description": ""
        }
        
        try:
            # Get the root element (could be form or entire page)
            elements = parsed_form.get("elements")
            if not elements:
                return context
            
            # Find the root document element
            root = elements
            while root.parent:
                root = root.parent
            
            # Extract page title from <title> tag
            title_tag = root.find("title")
            if title_tag:
                context["page_title"] = title_tag.get_text(strip=True)
            
            # Extract main heading from h1, h2, etc.
            for tag in ["h1", "h2", "h3"]:
                heading = root.find(tag)
                if heading:
                    heading_text = heading.get_text(strip=True)
                    if heading_text:
                        context["main_heading"] = heading_text
                        break
            
            # Extract form title from fieldset legend or form heading
            form_element = parsed_form.get("elements")
            if hasattr(form_element, 'find'):
                # Look for fieldset legend
                legend = form_element.find("legend")
                if legend:
                    context["form_title"] = legend.get_text(strip=True)
                
                # Look for headings within form
                for tag in ["h1", "h2", "h3", "h4"]:
                    form_heading = form_element.find(tag)
                    if form_heading:
                        form_heading_text = form_heading.get_text(strip=True)
                        if form_heading_text and not context["form_title"]:
                            context["form_title"] = form_heading_text
                        break

            # Extract description from p tags or intro text
            description_candidates = []
            if hasattr(form_element, 'find_all'):
                for p_tag in form_element.find_all("p"):
                    p_text = p_tag.get_text(strip=True)
                    if p_text and len(p_text) > 20:  # Meaningful description
                        description_candidates.append(p_text)

                    if description_candidates:
                        # Use the first meaningful description
                        context["description"] = description_candidates[0][:200]  # Limit length
            
        except Exception as e:
            print(f"DEBUG: Error extracting page context: {str(e)}")
        
        return context

    def _find_general_question_for_radio_checkbox(self, field: Dict[str, Any]) -> str:
        """Find the general question (fieldset legend) for radio/checkbox fields"""
        field_name = field.get("name", "")
        field_selector = field.get("selector", "")

        try:
            import re
            from bs4 import BeautifulSoup

            # Try to extract the question from the HTML context if available
            if hasattr(self, '_page_context') and self._page_context:
                form_html = getattr(self, '_current_form_html', None)
                if form_html:
                    soup = BeautifulSoup(form_html, 'html.parser')

                    # Strategy 1: Look for fieldset legend that contains this field
                    fieldsets = soup.find_all('fieldset')
                    for fieldset in fieldsets:
                        # Check if this fieldset contains our field
                        field_inputs = fieldset.find_all('input', {'name': field_name})
                        if field_inputs:
                            legend = fieldset.find('legend')
                            if legend:
                                legend_text = legend.get_text(strip=True)
                                if legend_text and len(legend_text) > 3:
                                    print(f"DEBUG: Found fieldset legend for {field_name}: '{legend_text}'")
                                    return legend_text

                    # Strategy 2: Look for heading or label before the field group
                    if field_name:
                        first_field = soup.find('input', {'name': field_name})
                        if first_field:
                            # Look for preceding h1-h6 or strong text
                            current = first_field
                            for _ in range(5):  # Look up to 5 levels up
                                parent = current.find_parent() if current else None
                                if not parent:
                                    break

                                # Look for headings in this container
                                for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'b']:
                                    heading = parent.find(tag)
                                    if heading:
                                        heading_text = heading.get_text(strip=True)
                                        # Check if this heading is related to email/contact
                                        if (heading_text and len(heading_text) > 5 and
                                                any(keyword in heading_text.lower() for keyword in
                                                    ['email', 'contact', 'address', 'phone', 'telephone', 'another'])):
                                            print(f"DEBUG: Found related heading for {field_name}: '{heading_text}'")
                                            return heading_text

                                current = parent

                    # Strategy 3: Look for descriptive text near the field
                    if field_name:
                        first_field = soup.find('input', {'name': field_name})
                        if first_field:
                            # Look for preceding p, div, span with descriptive text
                            for sibling in first_field.find_all_previous(['p', 'div', 'span', 'label']):
                                text = sibling.get_text(strip=True)
                                if (text and 20 < len(text) < 150 and  # Reasonable question length
                                        any(keyword in text.lower() for keyword in
                                            ['email', 'contact', 'address', 'phone', 'telephone', 'another', 'do you',
                                             'have you'])):
                                    print(f"DEBUG: Found descriptive text for {field_name}: '{text}'")
                                    return text

            # Generate a reasonable question from field name if HTML parsing failed
            if field_name:
                # Convert field name to readable question
                question = field_name.replace("_", " ").replace("-", " ").replace(".", " ")
                question = re.sub(r'([a-z])([A-Z])', r'\1 \2', question)  # Handle camelCase
                question = " ".join(word.capitalize() for word in question.split())

                # Enhanced question format for common patterns
                if any(keyword in question.lower() for keyword in ['email', 'address']):
                    if 'another' in question.lower() or 'additional' in question.lower():
                        return f"Do you have another email address?"
                    else:
                        return f"Please provide your email address"
                elif any(keyword in question.lower() for keyword in ['contact', 'phone', 'telephone']):
                    return f"Are you able to be contacted by telephone?"
                elif any(keyword in question.lower() for keyword in ['prefer', 'choice', 'select']):
                    return f"What is your {question.lower()}?"
                elif 'reason' in question.lower():
                    return f"Please explain the {question.lower()}"
                else:
                    return f"Please select your {question.lower()}"

            # Fallback
            return "Please make a selection"

        except Exception as e:
            print(f"DEBUG: Error finding general question: {e}")
            return field_name.replace("_", " ").title() if field_name else "Please make a selection"

    def _generate_field_question(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a question for a form field using page context"""
        field_name = field.get("name", "")
        field_label = field.get("label", "")
        field_type = field.get("type", "")
        placeholder = field.get("placeholder", "")
        
        # Generate question text using field-specific information first
        question_text = ""

        # Special handling for radio/checkbox: need to find the general question, not option-specific text
        if field_type in ["radio", "checkbox"]:
            # For radio/checkbox, field_label is the specific option text
            # We need to find the general question (fieldset legend)
            question_text = self._find_general_question_for_radio_checkbox(field)

            # If we couldn't find a general question, fallback to field name
            if not question_text or question_text == "Please make a selection":
                if field_name:
                    question_text = field_name.replace("_", " ").replace("-", " ").title()
                else:
                    question_text = f"{field_type} selection"
        else:
            # For other field types, use the standard strategy
            # Strategy 1: Use field label directly (most specific)
            if field_label:
                question_text = field_label.strip()

            # Strategy 2: Use placeholder if no label
            elif placeholder:
                question_text = placeholder.strip()

            # Strategy 3: Generate from field name
            elif field_name:
                question_text = field_name.replace("_", " ").replace("-", " ").title()

            # Strategy 4: Fallback to field type
            else:
                question_text = f"{field_type} field"
        
        # Clean up the question text
        question_text = question_text.strip()
        
        # Remove common HTML artifacts and clean up
        import re
        question_text = re.sub(r'\s+', ' ', question_text)  # Multiple spaces to single
        question_text = re.sub(r'[\r\n\t]', ' ', question_text)  # Remove line breaks
        
        # Ensure it ends properly (but don't force question marks for statements)
        if question_text and not question_text.endswith(('.', '?', ':')):
            # Only add question mark if it's clearly a question
            if any(word in question_text.lower() for word in ['what', 'which', 'how', 'when', 'where', 'who', 'choose', 'select', 'enter']):
                if not question_text.endswith('?'):
                    question_text += '?'
        
        return {
            "id": f"q_{field_name}_{uuid.uuid4().hex[:8]}",
            "field_selector": field["selector"],
            "field_name": field_name,
            "field_type": field_type,
            "field_label": field_label,
            "question": question_text,
            "required": field.get("required", False),
            "options": field.get("options", [])
        }
    
    def _generate_ai_answer(self, question: Dict[str, Any], profile: Dict[str, Any], profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate AI answer for a field question - Modified to only use external dummy data"""
        try:
            # First attempt: try to answer with profile_data (fill_data), with dummy data as secondary context
            primary_result = self._try_answer_with_data(question, profile, "profile_data", profile_dummy_data)

            # If we have good confidence and answer from profile data, use it
            if (primary_result["confidence"] >= 20 and  # Lowered threshold to prioritize real data
                primary_result["answer"] and 
                not primary_result["needs_intervention"]):
                print(
                    f"DEBUG: Using profile_data for field {question['field_name']}: answer='{primary_result['answer']}', confidence={primary_result['confidence']}")
                primary_result["used_dummy_data"] = False
                return primary_result

            # Second attempt: if profile_dummy_data is available and primary answer failed, try external dummy data
            if profile_dummy_data:
                print(
                    f"DEBUG: Trying external dummy data for field {question['field_name']} - primary confidence: {primary_result['confidence']}")
                print(f"DEBUG: Profile dummy data keys: {list(profile_dummy_data.keys())}")
                print(f"DEBUG: Profile dummy data full content: {json.dumps(profile_dummy_data, indent=2)}")
                dummy_result = self._try_answer_with_data(question, profile_dummy_data, "profile_dummy_data", profile)
                print(
                    f"DEBUG: Dummy data result - answer: '{dummy_result['answer']}', confidence: {dummy_result['confidence']}, needs_intervention: {dummy_result['needs_intervention']}")

                # Use dummy data if it's better than primary result OR if primary result has very low confidence
                if (dummy_result["confidence"] > primary_result["confidence"] or
                    (primary_result["confidence"] <= 10 and dummy_result["confidence"] >= 30)) and \
                        dummy_result["answer"] and not dummy_result["needs_intervention"]:
                    dummy_result["used_dummy_data"] = True
                    dummy_result["dummy_data_source"] = "profile_dummy_data"
                    print(
                        f"DEBUG: ✅ Using external dummy data for field {question['field_name']}: answer='{dummy_result['answer']}', confidence={dummy_result['confidence']}")
                    return dummy_result
                else:
                    print(
                        f"DEBUG: ❌ Not using dummy data - dummy confidence: {dummy_result['confidence']}, primary confidence: {primary_result['confidence']}")

            # If primary result has some reasonable confidence, use it even if not perfect
            if primary_result["confidence"] >= 10 and primary_result["answer"]:
                print(
                    f"DEBUG: Using profile_data despite lower confidence for field {question['field_name']}: answer='{primary_result['answer']}', confidence={primary_result['confidence']}")
                primary_result["used_dummy_data"] = False
                return primary_result

            # If all attempts failed, return empty result (no AI generation)
            print(f"DEBUG: No suitable data found for field {question['field_name']}, leaving empty")
            return {
                "question_id": question.get("id", ""),
                "field_selector": question.get("field_selector", ""),
                "field_name": question.get("field_name", ""),
                "answer": "",  # Leave empty if no data available
                "confidence": 0,
                "reasoning": "No suitable data found in profile_data or profile_dummy_data",
                "needs_intervention": True,  # Mark as needing intervention
                "used_dummy_data": False
            }
                
        except Exception as e:
            return {
                "question_id": question.get("id", ""),
                "field_selector": question.get("field_selector", ""),
                "field_name": question.get("field_name", ""),
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error generating answer: {str(e)}",
                "needs_intervention": True,
                "used_dummy_data": False
            }

    def _generate_smart_dummy_data(self, question: Dict[str, Any], profile: Dict[str, Any], primary_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intelligent dummy data using LLM"""
        try:
            field_name = question.get("field_name", "")
            field_type = question.get("field_type", "")
            field_label = question.get("field_label", "")
            options = question.get("options", [])

            # Create context for the LLM
            context = f"""
            Field Name: {field_name}
            Field Type: {field_type}
            Field Label: {field_label}
            """

            if options:
                context += f"\nAvailable Options: {[opt.get('value', opt.get('text', '')) for opt in options]}"

            # Special handling for checkbox fields that should be mutually exclusive
            is_mutually_exclusive_checkbox = (
                    field_type == "checkbox" and
                    any(keyword in field_name.lower() or keyword in field_label.lower()
                        for keyword in ["purpose", "type", "category", "status", "gender", "title"])
            )
            
            prompt = f"""
            Generate appropriate dummy data for this form field:
            
            {context}
            
            Requirements:
            1. Generate realistic, appropriate dummy data
            2. For sensitive fields (bank account, passport, SSN, credit card), return empty string
            3. For contact info, use realistic but fake data (e.g., "555-123-4567" for phone)
            4. For names, use common placeholder names
            5. For addresses, use realistic but generic addresses
            6. For dates, use reasonable dates (e.g., birth date should be adult age)
            
            {"7. IMPORTANT: This appears to be a mutually exclusive checkbox group. Select ONLY ONE option, not multiple." if is_mutually_exclusive_checkbox else ""}
            
            Return JSON in this exact format:
            {{
                "answer": "your_generated_value",
                "confidence": 85,
                "reasoning": "why this value was chosen"
            }}
            
            For radio/checkbox with options, return the option value, not the display text.
            {"For mutually exclusive checkboxes, return only ONE value." if is_mutually_exclusive_checkbox else ""}
            """

            # Use LLM call with workflow_id for logging
            if self._current_workflow_id:
                response = self._invoke_llm([HumanMessage(content=prompt)], self._current_workflow_id)
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])

            # Parse the response using robust JSON parsing
            try:
                result = robust_json_parse(response.content)

                # Ensure answer is a string, not a list
                if isinstance(result.get("answer"), list):
                    if is_mutually_exclusive_checkbox:
                        # For mutually exclusive checkboxes, take only the first item
                        result["answer"] = str(result["answer"][0]) if result["answer"] else ""
                    else:
                        # For regular checkboxes, join with commas
                        result["answer"] = ",".join(str(item) for item in result["answer"])
                else:
                    result["answer"] = str(result.get("answer", ""))

                # Validate the generated data
                if field_type in ["radio", "checkbox"] and options:
                    # Check if the answer matches any available option
                    answer_values = [v.strip() for v in result["answer"].split(",") if v.strip()]
                    valid_option_values = [str(opt.get("value", "")).lower() for opt in options]

                    # Filter to only include valid options
                    valid_answers = []
                    for answer_val in answer_values:
                        if answer_val.lower() in valid_option_values:
                            valid_answers.append(answer_val)

                    if valid_answers:
                        if is_mutually_exclusive_checkbox:
                            # For mutually exclusive, take only the first valid option
                            result["answer"] = valid_answers[0]
                        else:
                            # For regular checkboxes, join all valid options
                            result["answer"] = ",".join(valid_answers)
                    else:
                        # If no valid options found, pick the first available option
                        if options:
                            result["answer"] = str(options[0].get("value", ""))
                            result["reasoning"] = "Selected first available option as fallback"

                # CRITICAL: Add required fields for AI answer format
                final_result = {
                    "question_id": question.get("id", ""),
                    "field_selector": question.get("field_selector", ""),
                    "field_name": question.get("field_name", ""),
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 70),
                    "reasoning": result.get("reasoning", "Generated dummy data"),
                    "needs_intervention": False,  # AI provided an answer, no intervention needed
                    "used_dummy_data": True,  # This will be set by the caller
                    "dummy_data_source": "ai_generated"  # This will be set by the caller
                }

                print(
                    f"DEBUG: Smart dummy data generated for {field_name}: answer='{final_result['answer']}', confidence={final_result['confidence']}")

                return final_result

            except Exception as parse_error:
                print(f"DEBUG: Smart dummy data JSON parse error: {str(parse_error)}")
                # Fallback to basic dummy data generation
                return self._generate_basic_dummy_data(question)
                
        except Exception as e:
            print(f"DEBUG: Smart dummy data generation error: {str(e)}")
            # Fallback to basic dummy data generation
            return self._generate_basic_dummy_data(question)

    def _generate_basic_dummy_data(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic dummy data as fallback"""
        field_type = question.get("field_type", "")
        field_name = question.get("field_name", "").lower()
        options = question.get("options", [])

        # Basic dummy data patterns
        if "email" in field_name:
            answer = "user@example.com"
            confidence = 70
            reasoning = "Basic email dummy data"
        elif "phone" in field_name or "telephone" in field_name:
            answer = "555-123-4567"
            confidence = 70
            reasoning = "Basic phone dummy data"
        elif "name" in field_name:
            if "first" in field_name:
                answer = "John"
                confidence = 70
                reasoning = "Basic first name dummy data"
            elif "last" in field_name:
                answer = "Doe"
                confidence = 70
                reasoning = "Basic last name dummy data"
            else:
                answer = "John Doe"
                confidence = 70
                reasoning = "Basic name dummy data"
        elif field_type in ["radio", "checkbox"] and options:
            # For radio/checkbox, select the first option
            answer = options[0].get("value", "") if options else ""
            confidence = 60
            reasoning = "Selected first available option"
        elif field_type == "number":
            answer = "123"
            confidence = 60
            reasoning = "Basic number dummy data"
        else:
            answer = ""
            confidence = 30
            reasoning = "No appropriate dummy data pattern found"

        # Return in the correct format with all required fields
        return {
            "question_id": question.get("id", ""),
            "field_selector": question.get("field_selector", ""),
            "field_name": question.get("field_name", ""),
            "answer": answer,
            "confidence": confidence,
            "reasoning": reasoning,
            "needs_intervention": answer == "",  # Only need intervention if no answer generated
            "used_dummy_data": True,  # This will be set by the caller
            "dummy_data_source": "ai_generated"  # This will be set by the caller
        }

    def _try_answer_with_data(self, question: Dict[str, Any], data_source: Dict[str, Any], source_name: str,
                              secondary_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Try to answer a question using a specific data source, with optional secondary data for context"""
        try:
            # Create prompt for AI
            prompt = f"""
            # Role 
            You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
            analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
            filling in the form content, and returning it to the front end in a fixed json format.

            # Task
            Based on the user data from {source_name}, determine the appropriate value for this form field:
            
            Field Name: {question['field_name']}
            Field Type: {question['field_type']}
            Field Label: {question['field_label']}
            Field Selector: {question['field_selector']}
            Required: {question['required']}
            Question: {question['question']}
            Available Options: {json.dumps(question.get('options', []), indent=2) if question.get('options') else "No specific options provided"}
            
            ⚠️⚠️⚠️ CRITICAL REMINDER ⚠️⚠️⚠️: If Available Options shows multiple options above, you MUST analyze ALL of them!
            
            # Primary Data Source ({source_name}):
            {json.dumps(data_source, indent=2)}
            
            # Secondary Data Source (for context and cross-reference):
            {json.dumps(secondary_data, indent=2) if secondary_data else "None available"}
            
            MANDATORY FIRST STEP: COMPREHENSIVE ANALYSIS
            Before attempting to answer, you MUST complete these steps IN ORDER:
            
            STEP 1 - JSON DATA ANALYSIS:
            1. List ALL top-level fields in BOTH the Primary and Secondary data sources above
            2. List ALL nested fields (go deep into objects and arrays) in BOTH data sources
            3. Identify any fields that could semantically relate to the question in EITHER data source
            4. Pay special attention to boolean fields (has*, is*, can*, allow*, enable*) in BOTH sources
            5. Look for email-related fields (hasOtherEmail*, additionalEmail*, secondaryEmail*) in BOTH sources
            6. Cross-reference between Primary and Secondary data sources for comprehensive analysis
            
            STEP 2 - OPTION ANALYSIS (if Available Options are provided):
            1. List ALL available options (both text and value)
            2. For each option, analyze what type of data it expects (boolean, numerical range, text, etc.)
            3. For numerical options (like "3 years or less", "More than 3 years"), identify the threshold values
            4. Determine which option(s) could match the data you found in Step 1
            5. CRITICAL: Your final answer must be the option text or value, NOT the original data value
            
            # CRITICAL INSTRUCTIONS - DEEP JSON ANALYSIS AND SEMANTIC UNDERSTANDING:
             1. **COMPREHENSIVE JSON ANALYSIS**: FIRST, carefully read and analyze the ENTIRE JSON structure. List all available fields and their values before attempting to answer
             2. **SEMANTIC UNDERSTANDING**: Understand the MEANING of each data field, not just the field name
             3. **INTELLIGENT FIELD DISCOVERY**: Look for fields that semantically match the question, even if field names don't exactly match:
                - For "another email" questions: Look for "hasOtherEmail*", "additionalEmail*", "secondaryEmail*", "otherEmail*"
                - For "contact by phone" questions: Look for "canContact*", "allowContact*", "phoneContact*"
                - For any boolean question: Look for "has*", "is*", "can*", "allow*", "enable*" fields
                - For "visa length" questions: Look for "visaLength", "applicationDetails", "duration", "period"
             4. **SEARCH ALL DATA**: Thoroughly search through ALL nested objects and arrays - DO NOT SKIP ANY LEVELS
             5. **SMART FIELD MAPPING EXAMPLES**:
               - Field about "another email" + Data "hasOtherEmailAddresses: false" → Answer: "false/no" (HIGH confidence 85+)
               - Field about "telephone contact" + Data "contactInformation.telephoneNumber" → Use the phone number
               - Field about "telephone type" + Data "contactInformation.telephoneType: 'Mobile'" → Answer: "mobile"
               - Field about "name" + Data "personalDetails.givenName: 'John'" → Answer: "John"
               - Field about "email" + Data "contactInformation.primaryEmail" → Use the email address
               - Field about "birth date" + Data "personalDetails.dateOfBirth" → Use the date
               - Field about "visa length" + Data "applicationDetails.visaLength: '5 years'" → Answer: "5 years" or match to appropriate option
            5. **BOOLEAN FIELD INTELLIGENCE**: 
               - For yes/no questions, understand boolean values: true="yes", false="no"
               - Field asking "Do you have X?" + Data "hasX: false" → Answer: "false" or "no" (confidence 85+)
               - Field asking "Are you Y?" + Data "isY: true" → Answer: "true" or "yes" (confidence 85+)
            6. **NUMERICAL COMPARISON AND RANGE MATCHING**:
               - For duration/length questions, compare numerical values intelligently:
               - "5 years" vs "3 years or less" → Does NOT match (5 > 3)
               - "5 years" vs "More than 3 years" → MATCHES (5 > 3) (confidence 90+)
               - "2 years" vs "3 years or less" → MATCHES (2 ≤ 3) (confidence 90+)
               - "2 years" vs "More than 3 years" → Does NOT match (2 ≤ 3)
               - Extract numbers from text and perform logical comparisons
            7. **SEMANTIC MATCHING PATTERNS**:
               - "another/additional/other email" matches "hasOtherEmailAddresses", "additionalEmail", "secondaryEmail"
               - "telephone/phone number" matches "telephoneNumber", "phoneNumber", "contactNumber"
               - "first/given name" matches "givenName", "firstName", "name"
               - "visa length/duration" matches "visaLength", "duration", "period"
               - Use SEMANTIC UNDERSTANDING, not just string matching
            8. **COMPREHENSIVE OPTION MATCHING**: For radio/checkbox fields with multiple options:
               - MANDATORY: Check data value against ALL available options, not just the first one
               - Use logical comparison (numerical, boolean, string matching)
               - Example: Data "5 years" should be checked against both "3 years or less" AND "More than 3 years"
               - CRITICAL: Your answer must be one of the available option values or option text, not the original data value
               - For numerical ranges: "5 years" + options ["3 years or less", "More than 3 years"] → Answer: "More than 3 years" (NOT "5 years")
            9. **CONFIDENCE SCORING - FAVOR SEMANTIC UNDERSTANDING**: 
               - 90-100: Perfect semantic match (hasOtherEmailAddresses:false for "Do you have another email" question, or "5 years" for "More than 3 years")
               - 80-89: Strong semantic match with clear meaning
               - 70-79: Good semantic inference from data structure
               - 50-69: Reasonable inference from context
               - 30-49: Weak match, uncertain
               - 0-29: No suitable semantic match found
            10. **VALIDATION**: Ensure the data type and format matches the field requirements
            11. **MUTUAL EXCLUSIVITY**: For radio/checkbox fields, select only ONE appropriate value
            12. **EMPTY DATA HANDLING**: If data exists but is empty/null, set needs_intervention=true
            
            # Special Instructions for Telephone Fields:
            - For "telephoneNumber": Extract the actual phone number from contactInformation.telephoneNumber
            - For "telephoneNumberType": Map from contactInformation.telephoneType ("Mobile" -> "mobile", "Home" -> "home", "Business" -> "business")
            - For "telephoneNumberPurpose": If from contactInformation, likely "useInUK" (give high confidence)
            
            # SEMANTIC MATCHING EXAMPLES (CRITICAL - STUDY THESE PATTERNS):
            
            ## BOOLEAN/YES-NO EXAMPLES:
            - Question "Do you have another email address?" + Data "hasOtherEmailAddresses: false" = Answer: "false" (confidence 90+)
            - Question "Do you have another email address?" + Data "hasOtherEmailAddresses: true" = Answer: "true" (confidence 90+)
            - Question asking about additional/other/secondary email + ANY field containing "hasOther*", "additional*", "secondary*" → Use that boolean value
            - Question about "contact by phone" + Data "canContactByPhone: false" = Answer: "false" (confidence 90+)
            
            ## DIRECT TEXT EXAMPLES:
            - Field "telephoneNumber" + Data "contactInformation.telephoneNumber: '+1234567890'" = Answer: "+1234567890" (confidence 95)
            - Field "telephoneNumberType" + Data "contactInformation.telephoneType: 'Mobile'" = Answer: "mobile" (confidence 90)
            - Field "givenName" + Data "personalDetails.givenName: 'John'" = Answer: "John" (confidence 95)
            
            ## NUMERICAL RANGE EXAMPLES (MOST IMPORTANT FOR YOUR CASE):
            - Question "What is the length of the visa?" + Data "visaLength: '5 years'" + Options ["3 years or less", "More than 3 years"]:
              * Step 1: Found data "5 years"
              * Step 2: Check against "3 years or less" → 5 > 3, so NO MATCH
              * Step 3: Check against "More than 3 years" → 5 > 3, so MATCH!
              * Final Answer: "More than 3 years" (confidence 95)
            - Question "What is the length of the visa?" + Data "visaLength: '2 years'" + Options ["3 years or less", "More than 3 years"]:
              * Step 1: Found data "2 years"  
              * Step 2: Check against "3 years or less" → 2 ≤ 3, so MATCH!
              * Final Answer: "3 years or less" (confidence 95)
            
            ## KEY PRINCIPLE: 
            When options are provided, your answer MUST be one of the option texts/values, NOT the original data value!
            
            ## CRITICAL FINAL STEP - ANSWER VALIDATION:
            Before providing your final JSON response, you MUST:
            1. Re-check that your "answer" field contains EXACTLY one of the available option values or texts
            2. If you found data like "5 years" and determined it matches "More than 3 years", your answer MUST be "More than 3 years" or "moreThanThreeYears"
            3. NEVER put the original data value (like "5 years") in the answer field when options are provided
            4. Double-check your logic: if your reasoning says option X matches, your answer MUST be option X
            
            # Response Format (JSON only):
            {{
                "answer": "your answer here or empty if no data available",
                "confidence": 85,
                "reasoning": "detailed explanation of data source and matching logic - BE SPECIFIC about which data field you used",
                "needs_intervention": false
            }}
            
            # Examples of HIGH confidence scenarios (70+):
            - Field "telephoneNumber" matches contactInformation.telephoneNumber
            - Field "givenName" matches personalDetails.givenName
            - Field "email" matches contactInformation.primaryEmail
            - Field "telephoneNumberType" can be mapped from contactInformation.telephoneType
            
            # Examples of when needs_intervention should be true:
            - No matching data found in any nested structure
            - Data exists but is empty/null/undefined
            - Field is required but confidence is below 30
            - Data type mismatch that cannot be resolved
            
            # Examples of when needs_intervention should be false:
            - Exact match found in data source (high confidence 70+)
            - Semantic match found in data source (medium confidence 50+)
            - Data type matches field requirements
            - Data is empty/null/undefined but field not required
            
            REMEMBER: Your goal is to USE the real data provided whenever possible. Be generous with confidence scores for real data matches!
            """

            response = self._invoke_llm([HumanMessage(content=prompt)])
            
            try:
                # Try to parse JSON response using robust parsing
                result = robust_json_parse(response.content)
                
                # Determine if intervention is needed based on AI response and confidence
                needs_intervention = result.get("needs_intervention", False)
                confidence = result.get("confidence", 0)
                answer = result.get("answer", "")

                # Normalize answer to string format
                if isinstance(answer, list):
                    # Special handling for checkbox fields that should be mutually exclusive
                    field_name = question.get("field_name", "").lower()
                    field_label = question.get("field_label", "").lower()
                    is_mutually_exclusive = any(keyword in field_name or keyword in field_label
                                                for keyword in ["purpose", "type", "category", "status"])

                    if is_mutually_exclusive and len(answer) > 1:
                        # For mutually exclusive fields, take only the first value
                        answer = str(answer[0]) if answer else ""
                        print(
                            f"DEBUG: _try_answer_with_data - Converted mutually exclusive list to single value: {answer}")
                    else:
                        # For regular checkboxes, join with commas
                        answer = ",".join(str(item) for item in answer) if answer else ""
                elif answer is None:
                    answer = ""
                else:
                    answer = str(answer)
                
                # Additional logic: if confidence is very low or answer is empty for required fields
                if not needs_intervention:
                    if question.get("required", False) and not answer:
                        needs_intervention = True
                    elif confidence < 30:  # Very low confidence threshold
                        needs_intervention = True
                
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "field_type": question.get("field_type", ""),
                    "field_label": question.get("field_label", ""),
                    "options": question.get("options", []),
                    "answer": answer,
                    "confidence": confidence,
                    "reasoning": result.get("reasoning", ""),
                    "needs_intervention": needs_intervention
                }
            except Exception as parse_error:
                # Fallback if even robust parsing fails
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "field_type": question.get("field_type", ""),
                    "field_label": question.get("field_label", ""),
                    "options": question.get("options", []),
                    "answer": "",
                    "confidence": 0,
                    "reasoning": f"JSON parsing failed: {str(parse_error)}",
                    "needs_intervention": True
                }
                
        except Exception as e:
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "field_type": question.get("field_type", ""),
                "field_label": question.get("field_label", ""),
                "options": question.get("options", []),
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error with {source_name}: {str(e)}",
                "needs_intervention": True
            }

    def _generate_form_action(self, merged_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate form action from merged question-answer data - ALWAYS GENERATES ACTION"""
        # Get answer and confidence
        answer = merged_data.get("answer", "")
        confidence = merged_data.get("confidence", 0)

        # CRITICAL: Always generate action with selector and type
        # Even if no answer, we need to provide the basic action structure
        
        # Get field information
        selector = merged_data["field_selector"]
        field_name = merged_data["field_name"]
        field_type = merged_data.get("field_type", "")
        options = merged_data.get("options", [])

        actions = []

        # Determine action type and value based on field type
        if field_type == "radio":
            # For radio buttons, generate click action
            if answer:
                # Update selector to include the specific value
                if "[value='" not in selector and options:
                    # Find the matching option
                    for option in options:
                        if option.get("value") == answer or option.get("text") == answer:
                            updated_selector = selector.replace("]", f"][value='{option.get('value')}']")
                            actions.append({
                                "selector": updated_selector,
                                "type": "click",
                                "value": option.get('value', answer)
                            })
                            break
                else:
                    actions.append({
                        "selector": selector,
                        "type": "click",
                        "value": answer
                    })
            else:
                # No answer - generate base action for first option or default
                if options:
                    first_option = options[0]
                    updated_selector = selector.replace("]", f"][value='{first_option.get('value')}']")
                    actions.append({
                        "selector": updated_selector,
                        "type": "click",
                        "value": ""  # Empty value indicates no selection yet
                    })
                else:
                    actions.append({
                        "selector": selector,
                        "type": "click",
                        "value": ""
                    })
                
        elif field_type == "checkbox":
            # For checkboxes, handle multiple values separated by commas
            if answer:
                values = [v.strip() for v in answer.split(",") if v.strip()]

                for value in values:
                    # Create selector for this specific checkbox value
                    if "[value='" not in selector and options:
                        # Find the matching option
                        for option in options:
                            if option.get("value") == value or option.get("text") == value:
                                # Build selector with specific value
                                updated_selector = selector.replace("]", f"][value='{option.get('value')}']")
                                actions.append({
                                    "selector": updated_selector,
                                    "type": "click",
                                    "value": option.get('value', value)
                                })
                                break
                    else:
                        # Use the selector as-is if it already includes value
                        actions.append({
                            "selector": selector,
                            "type": "click",
                            "value": value
                        })
            else:
                # No answer - generate base action for first option or default
                if options:
                    first_option = options[0]
                    updated_selector = selector.replace("]", f"][value='{first_option.get('value')}']")
                    actions.append({
                        "selector": updated_selector,
                        "type": "click",
                        "value": ""  # Empty value indicates no selection yet
                    })
                else:
                    actions.append({
                        "selector": selector,
                        "type": "click",
                        "value": ""
                    })

        elif field_type in ["text", "email", "password", "number", "tel", "url", "date", "time", "datetime-local"]:
            # For input fields, use input action with value (even if empty)
            actions.append({
                "selector": selector,
                "type": "input",
                "value": answer if answer else ""
            })
            
        elif field_type == "select":
            # For select dropdowns, use input action with value (even if empty)
            actions.append({
                "selector": selector,
                "type": "input",
                "value": answer if answer else ""
            })

        elif field_type == "textarea":
            # For textarea, use input action with value (even if empty)
            actions.append({
                "selector": selector,
                "type": "input",
                "value": answer if answer else ""
            })
        else:
            # Default case - always generate input action with selector and type
            actions.append({
                "selector": selector,
                "type": "input",
                "value": answer if answer else ""
            })

        # CRITICAL: Always return actions list, never None
        # Even if no answer, we still provide the action structure
        return actions if actions else [{
            "selector": selector,
            "type": "input",
            "value": ""
        }]

    def _find_answer_for_question(self, question: Dict[str, Any], answers: List[Dict[str, Any]]) -> Optional[
        Dict[str, Any]]:
        """Find the answer for a given question"""
        question_selector = question.get("field_selector", "")
        question_id = question.get("id", "")
        question_field_name = question.get("field_name", "")
        
        for answer in answers:
            # Try multiple matching strategies
            # Strategy 1: Match by field_selector (if both have it)
            answer_selector = answer.get("field_selector", "")
            if question_selector and answer_selector and answer_selector == question_selector:
                return answer

            # Strategy 2: Match by question_id (if answer has it)
            answer_question_id = answer.get("question_id", "")
            if question_id and answer_question_id and answer_question_id == question_id:
                return answer

            # Strategy 3: Match by field_name (if both have it)
            answer_field_name = answer.get("field_name", "")
            if question_field_name and answer_field_name and answer_field_name == question_field_name:
                return answer
        
        return None

    def _is_duration_comparison_match(self, ai_value: str, option_text: str) -> bool:
        """Check if AI value matches option text through numerical comparison for duration fields"""
        try:
            import re

            # Extract numbers from AI value (e.g., "5 years" -> 5)
            ai_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:year|month|day)', ai_value.lower())
            if not ai_match:
                return False

            ai_number = float(ai_match.group(1))

            # Check different option patterns
            option_lower = option_text.lower()

            # Pattern: "3 years or less" / "X years or less"
            less_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:year|month|day)s?\s*or\s*less', option_lower)
            if less_match:
                threshold = float(less_match.group(1))
                return ai_number <= threshold

            # Pattern: "more than 3 years" / "more than X years"
            more_match = re.search(r'more\s*than\s*(\d+(?:\.\d+)?)\s*(?:year|month|day)', option_lower)
            if more_match:
                threshold = float(more_match.group(1))
                return ai_number > threshold

            # Pattern: "less than 3 years" / "less than X years"
            less_than_match = re.search(r'less\s*than\s*(\d+(?:\.\d+)?)\s*(?:year|month|day)', option_lower)
            if less_than_match:
                threshold = float(less_than_match.group(1))
                return ai_number < threshold

            # Pattern: "between X and Y years"
            between_match = re.search(r'between\s*(\d+(?:\.\d+)?)\s*and\s*(\d+(?:\.\d+)?)\s*(?:year|month|day)',
                                      option_lower)
            if between_match:
                min_val = float(between_match.group(1))
                max_val = float(between_match.group(2))
                return min_val <= ai_number <= max_val

            return False

        except (ValueError, AttributeError) as e:
            print(f"DEBUG: _is_duration_comparison_match - Error parsing values: {e}")
            return False

    def _create_answer_data(self, question: Dict[str, Any], ai_answer: Optional[Dict[str, Any]], needs_intervention: bool) -> List[Dict[str, Any]]:
        """Create answer data array based on field type and AI answer
        
        Updated logic:
        - If AI provided an answer (including dummy data): mark as check=1 and show the answer
        - If no answer at all: mark as check=0 for user selection
        - For radio/checkbox: use option text as name, not question text
        """
        field_type = question["field_type"]
        options = question.get("options", [])
        ai_answer_value = ai_answer.get("answer", "") if ai_answer else ""
        has_ai_answer = bool(ai_answer_value and ai_answer_value.strip())

        # Check if this is dummy data (AI generated or provided)
        is_dummy_data = ai_answer.get("used_dummy_data", False) if ai_answer else False

        print(f"DEBUG: _create_answer_data - Field: {question.get('field_name', 'unknown')}")
        print(f"DEBUG: _create_answer_data - AI answer: '{ai_answer_value}'")
        print(f"DEBUG: _create_answer_data - Has AI answer: {has_ai_answer}")
        print(f"DEBUG: _create_answer_data - Is dummy data: {is_dummy_data}")
        print(f"DEBUG: _create_answer_data - Needs intervention: {needs_intervention}")
        print(f"DEBUG: _create_answer_data - Available options: {[opt.get('text', '') for opt in options]}")
        print(
            f"DEBUG: _create_answer_data - AI reasoning: {ai_answer.get('reasoning', 'No reasoning') if ai_answer else 'No AI answer'}")
        
        if field_type in ["radio", "checkbox", "select"] and options:
            if has_ai_answer:  # AI provided an answer (real or dummy)
                # AI已回答：找到匹配的选项并使用选项文本
                matched_option = None

                # Enhanced matching logic for boolean/yes-no questions and numerical comparisons
                ai_value_lower = ai_answer_value.lower().strip()
                
                for option in options:
                    option_value = option.get("value", "").lower()
                    option_text = option.get("text", "").lower()

                    print(
                        f"DEBUG: _create_answer_data - Checking option: '{option.get('text', '')}' (value: '{option.get('value', '')}')")

                    # Direct string matching (exact match)
                    if (option_value == ai_value_lower or option_text == ai_value_lower):
                        matched_option = option
                        print(
                            f"DEBUG: _create_answer_data - Direct string match found: '{ai_answer_value}' matches '{option.get('text', '')}'")
                        break

                    # Boolean value mapping for yes/no questions
                    if ai_value_lower in ["false", "no", "0"]:
                        if option_value in ["false", "no", "0"] or option_text in ["no", "false", "否", "不是"]:
                            matched_option = option
                            print(
                                f"DEBUG: _create_answer_data - Boolean match found: '{ai_answer_value}' matches '{option.get('text', '')}'")
                            break
                    elif ai_value_lower in ["true", "yes", "1"]:
                        if option_value in ["true", "yes", "1"] or option_text in ["yes", "true", "是", "是的"]:
                            matched_option = option
                            print(
                                f"DEBUG: _create_answer_data - Boolean match found: '{ai_answer_value}' matches '{option.get('text', '')}'")
                            break

                # If no direct match found, try numerical comparison for duration/period questions
                if not matched_option:
                    for option in options:
                        option_text = option.get("text", "")
                        if self._is_duration_comparison_match(ai_answer_value, option_text):
                            matched_option = option
                            print(
                                f"DEBUG: _create_answer_data - Numerical match found: '{ai_answer_value}' matches '{option_text}'")
                            break

                if matched_option:
                    # 为radio/checkbox字段生成正确的选择器
                    if field_type in ["radio", "checkbox"]:
                        # 生成指向特定选项值的选择器
                        field_name = question.get("field_name", "")
                        option_value = matched_option.get("value", "")

                        # 🚀 OPTIMIZATION: 智能选择器生成
                        # 首先尝试使用字段名+值的ID格式（最常见的模式）
                        if option_value in ["true", "false"]:
                            # 尝试常见的ID模式
                            possible_selectors = [
                                f"#{field_name}_{option_value}",  # addAnother_false
                                f"#value_{option_value}",  # value_false
                                f"#{field_name}_{option_value.title()}",  # addAnother_False
                                f"input[type='{field_type}'][name='{field_name}'][value='{option_value}']"  # fallback
                            ]
                        else:
                            # 对于非布尔值，使用属性选择器
                            possible_selectors = [
                                f"#{field_name}_{option_value}",
                                f"input[type='{field_type}'][name='{field_name}'][value='{option_value}']"
                            ]

                        # 选择第一个可能的选择器作为主选择器
                        correct_selector = possible_selectors[0]
                    else:
                        correct_selector = question["field_selector"]

                    print(
                        f"DEBUG: _create_answer_data - Generated correct selector: '{correct_selector}' for option '{matched_option.get('text', '')}'")
                    
                    # 使用匹配选项的文本作为答案显示
                    return [{
                        "name": matched_option.get("text", matched_option.get("value", "")),
                        "value": matched_option.get("value", ""),
                        "check": 1,  # Mark as selected because AI provided an answer
                        "selector": correct_selector
                    }]
                else:
                    # 没找到匹配选项，选择第一个可用选项作为默认值（而不是使用AI原始答案）
                    if options:
                        default_option = options[0]

                        # 为默认选项生成正确的选择器
                        if field_type in ["radio", "checkbox"]:
                            field_name = question.get("field_name", "")
                            option_value = default_option.get("value", "")

                            if option_value in ["true", "false"]:
                                default_selector = f"#{field_name}_{option_value}"  # 使用字段名+值的格式
                            else:
                                default_selector = f"input[type='{field_type}'][name='{field_name}'][value='{option_value}']"
                        else:
                            default_selector = question["field_selector"]
                        
                        print(
                            f"DEBUG: _create_answer_data - No matching option found for AI answer '{ai_answer_value}', using first option: '{default_option.get('text', '')}' with selector '{default_selector}'")
                        return [{
                            "name": default_option.get("text", default_option.get("value", "")),
                            "value": default_option.get("value", ""),
                            "check": 1,  # Mark as selected because AI provided an answer
                            "selector": default_selector
                        }]
                    else:
                        # 没有选项，使用AI原始答案作为fallback
                        return [{
                            "name": ai_answer_value,
                            "value": ai_answer_value,
                            "check": 1,  # Mark as selected because AI provided an answer
                            "selector": question["field_selector"]
                        }]
            else:
                # 没有答案：存储完整选项列表供用户选择
                answer_data = []
                
                for option in options:
                    # 构建选择器
                    if field_type == "select":
                        selector = question["field_selector"]
                    else:
                        # 为radio/checkbox生成正确的选择器
                        field_name = question.get("field_name", "")
                        option_value = option.get("value", "")

                        # 使用字段名+值的ID格式（最常见的模式）
                        if option_value in ["true", "false"]:
                            selector = f"#{field_name}_{option_value}"  # addAnother_false
                        else:
                            # 使用属性选择器
                            selector = f"input[type='{field_type}'][name='{field_name}'][value='{option_value}']"
                    
                    answer_data.append({
                        "name": option.get("text", option.get("value", "")),  # 使用选项文本，不是问题文本
                        "value": option.get("value", ""),
                        "check": 0,  # Not selected - waiting for user input
                        "selector": selector
                    })
                
                return answer_data
        else:
            # For text inputs, textarea, etc.
            # If AI provided an answer (including dummy data), use it and mark as filled
            if has_ai_answer:
                return [{
                    "name": ai_answer_value,
                    "value": ai_answer_value,
                    "check": 1,  # Mark as filled because AI provided an answer
                    "selector": question["field_selector"]
                }]
            else:
                # No answer from AI - show empty field for user input
                return [{
                    "name": "",
                    "value": "",
                    "check": 0,  # Empty field - waiting for user input
                    "selector": question["field_selector"]
                }]

    async def analyze_step_async(self, html_content: str, workflow_id: str, current_step_key: str) -> Dict[str, Any]:
        """
        异步版本的步骤分析 - 分析当前页面属于当前步骤还是下一步骤
        
        改进逻辑：
        - 同时获取当前步骤和下一步骤的上下文信息
        - 使用 LLM 比较页面内容更符合哪个步骤
        - 如果属于下一步，完成当前步骤并激活下一步
        """
        try:
            # 提取页面问题
            page_analysis = self._extract_page_questions(html_content)

            # 获取当前步骤上下文
            current_step_context = self._get_step_context(workflow_id, current_step_key)

            # 获取下一步骤上下文
            next_step_key = self._find_next_step(workflow_id, current_step_key)
            next_step_context = None
            if next_step_key:
                next_step_context = self._get_step_context(workflow_id, next_step_key)

            # 使用 LLM 进行比较分析 (异步版本)
            analysis_result = await self._analyze_with_llm_async(
                page_analysis=page_analysis,
                current_step_context=current_step_context,
                next_step_context=next_step_context,
                current_step_key=current_step_key,
                next_step_key=next_step_key
            )

            # 如果页面属于下一步骤，执行步骤转换
            if analysis_result.get("belongs_to_next_step", False) and next_step_key:
                print(
                    f"[workflow_id:{workflow_id}] DEBUG: Page belongs to next step {next_step_key}, executing step transition")

                # 获取当前步骤实例
                current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
                if current_step:
                    # 1. 完成当前步骤
                    self.step_repo.update_step_status(current_step.step_instance_id, StepStatus.COMPLETED_SUCCESS)
                    current_step.completed_at = datetime.utcnow()
                    print(f"[workflow_id:{workflow_id}] DEBUG: Completed current step {current_step_key}")

                    # 2. 激活下一个步骤
                    next_step = self.step_repo.get_step_by_key(workflow_id, next_step_key)
                    if next_step:
                        self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
                        next_step.started_at = datetime.utcnow()
                        print(f"[workflow_id:{workflow_id}] DEBUG: Activated next step {next_step_key}")

                        # 3. 更新工作流实例的当前步骤
                        from src.database.workflow_repositories import WorkflowInstanceRepository
                        instance_repo = WorkflowInstanceRepository(self.db)
                        instance_repo.update_instance_status(
                            workflow_id,
                            None,  # 保持当前工作流状态
                            next_step_key  # 更新当前步骤键
                        )
                        print(f"[workflow_id:{workflow_id}] DEBUG: Updated workflow current step to {next_step_key}")

                        # 4. 更新分析结果，指示应该使用下一步骤执行
                        analysis_result.update({
                            "should_use_next_step": True,
                            "next_step_key": next_step_key,
                            "next_step_instance_id": next_step.step_instance_id,
                            "step_transition_completed": True
                        })

                # 提交数据库更改
                self.db.commit()
                print(f"[workflow_id:{workflow_id}] DEBUG: Step transition completed and committed")
            else:
                # 页面属于当前步骤，继续使用当前步骤
                analysis_result.update({
                    "should_use_next_step": False,
                    "step_transition_completed": False
                })

            return analysis_result

        except Exception as e:
            print(f"Error analyzing step: {str(e)}")
            # 发生错误时回滚数据库更改
            self.db.rollback()
            return {
                "belongs_to_current_step": True,
                "belongs_to_next_step": False,
                "should_use_next_step": False,
                "main_question": "",
                "step_transition_completed": False,
                "next_step_key": None,
                "reasoning": f"Error analyzing step: {str(e)}"
            }

    async def _analyze_with_llm_async(self, page_analysis: Dict[str, Any],
                                      current_step_context: Dict[str, Any],
                                      next_step_context: Optional[Dict[str, Any]],
                                      current_step_key: str,
                                      next_step_key: Optional[str]) -> Dict[str, Any]:
        """异步版本的 LLM 步骤分析"""
        try:
            # 构建分析提示
            prompt = f"""
            # Role
            You are a workflow step analyzer. Determine which step this page belongs to.
            
            # Task
            Analyze the current page content and determine if it belongs to the current step or the next step.
            
            # Page Analysis:
            {json.dumps(page_analysis, indent=2, ensure_ascii=False)}
            
            # Current Step Context ({current_step_key}):
            {json.dumps(current_step_context, indent=2, ensure_ascii=False)}
            
            # Next Step Context ({next_step_key if next_step_key else 'None'}):
            {json.dumps(next_step_context, indent=2, ensure_ascii=False) if next_step_context else 'No next step available'}
            
            # Instructions
            1. Compare the page content with both step contexts
            2. Determine which step the page content best matches
            3. Consider the questions being asked, form fields, and page title
            4. Provide confidence scores for your analysis
            
            # Response Format (JSON only):
            {{
                "belongs_to_current_step": true/false,
                "belongs_to_next_step": true/false,
                "main_question": "primary question or topic on the page",
                "confidence_current": 0-100,
                "confidence_next": 0-100,
                "reasoning": "detailed explanation of the analysis"
            }}
            
            **IMPORTANT**: Return ONLY the JSON response, no other text.
            """

            # 使用异步 LLM 调用
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            try:
                result = robust_json_parse(response.content)

                # 验证和设置默认值
                if not isinstance(result, dict):
                    raise ValueError("Response is not a valid JSON object")

                result.setdefault("belongs_to_current_step", True)
                result.setdefault("belongs_to_next_step", False)
                result.setdefault("main_question", "")
                result.setdefault("confidence_current", 50)
                result.setdefault("confidence_next", 0)
                result.setdefault("reasoning", "No reasoning provided")

                return result

            except Exception as parse_error:
                print(f"DEBUG: JSON parsing error in _analyze_with_llm_async: {str(parse_error)}")
                return {
                    "belongs_to_current_step": True,
                    "belongs_to_next_step": False,
                    "main_question": "",
                    "confidence_current": 50,
                    "confidence_next": 0,
                    "reasoning": f"Failed to parse LLM response: {str(parse_error)}"
                }

        except Exception as e:
            print(f"DEBUG: Error in _analyze_with_llm_async: {str(e)}")
            return {
                "belongs_to_current_step": True,
                "belongs_to_next_step": False,
                "main_question": "",
                "confidence_current": 50,
                "confidence_next": 0,
                "reasoning": f"Error during LLM analysis: {str(e)}"
            }


class LangGraphFormProcessor:
    """Form processor using LangGraph workflow"""
    
    def __init__(self, db_session: Session):
        """Initialize processor with database session"""
        self.db = db_session
        self.step_repo = StepInstanceRepository(db_session)  # 添加 StepInstanceRepository
        self.step_analyzer = StepAnalyzer(db_session)  # 添加 StepAnalyzer
        self.llm = ChatOpenAI(
            base_url=os.getenv("MODEL_BASE_URL"),
            api_key=SecretStr(os.getenv("MODEL_API_KEY")),
            model=os.getenv("MODEL_WORKFLOW_NAME"),
            temperature=0,
        )
        self._current_workflow_id = None  # Thread isolation for LLM calls

        # 🚀 OPTIMIZATION 3: Smart caching system
        self._field_analysis_cache = {}  # Cache for field analysis results
        self._batch_analysis_cache = {}  # Cache for batch analysis results
        self._cache_max_size = 100  # Maximum cache entries
        self._cache_ttl = 3600  # Cache TTL in seconds (1 hour)

        # Create the workflow app
        workflow = self._create_workflow()
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        import hashlib
        # Create a deterministic hash from the data
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False

        timestamp = cache_entry.get("timestamp", 0)
        current_time = time.time()
        return (current_time - timestamp) < self._cache_ttl

    def _get_from_cache(self, cache_key: str, cache_type: str = "field") -> Optional[Dict[str, Any]]:
        """Get result from cache if valid"""
        cache = self._field_analysis_cache if cache_type == "field" else self._batch_analysis_cache

        if cache_key in cache:
            entry = cache[cache_key]
            if self._is_cache_valid(entry):
                print(f"DEBUG: Cache HIT for {cache_type} analysis: {cache_key[:8]}...")
                return entry.get("result")
            else:
                # Remove expired entry
                del cache[cache_key]
                print(f"DEBUG: Cache EXPIRED for {cache_type} analysis: {cache_key[:8]}...")

        print(f"DEBUG: Cache MISS for {cache_type} analysis: {cache_key[:8]}...")
        return None

    def _save_to_cache(self, cache_key: str, result: Any, cache_type: str = "field"):
        """Save result to cache"""
        cache = self._field_analysis_cache if cache_type == "field" else self._batch_analysis_cache

        # Clean up cache if it's getting too large
        if len(cache) >= self._cache_max_size:
            # Remove oldest entries (simple FIFO strategy)
            oldest_keys = list(cache.keys())[:len(cache) - self._cache_max_size + 10]
            for key in oldest_keys:
                del cache[key]
            print(f"DEBUG: Cache cleanup - removed {len(oldest_keys)} old entries")

        cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }
        print(f"DEBUG: Cache SAVE for {cache_type} analysis: {cache_key[:8]}...")

    def set_workflow_id(self, workflow_id: str):
        """Set the current workflow ID for thread isolation"""
        self._current_workflow_id = workflow_id
        # Also set it for the step analyzer
        if hasattr(self.step_analyzer, 'set_workflow_id'):
            self.step_analyzer.set_workflow_id(workflow_id)

    def _invoke_llm(self, messages: List, workflow_id: str = None):
        """Invoke LLM (workflow_id kept for logging purposes only)"""
        # workflow_id is kept for logging but not used in LLM call
        used_workflow_id = workflow_id or self._current_workflow_id
        if used_workflow_id:
            print(f"[workflow_id:{used_workflow_id}] DEBUG: Invoking LLM")
        return self.llm.invoke(messages)
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(FormAnalysisState)
        
        # Add nodes
        workflow.add_node("html_parser", self._html_parser_node)
        workflow.add_node("field_detector", self._field_detector_node)
        workflow.add_node("question_generator", self._question_generator_node)
        workflow.add_node("profile_retriever", self._profile_retriever_node)
        workflow.add_node("ai_answerer", self._ai_answerer_node)
        workflow.add_node("qa_merger", self._qa_merger_node)  # New node for merging Q&A
        workflow.add_node("action_generator", self._action_generator_node)
        workflow.add_node("llm_action_generator", self._llm_action_generator_node)  # New LLM-based action generator
        workflow.add_node("result_saver", self._result_saver_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Set entry point
        workflow.set_entry_point("html_parser")
        
        # Add edges
        workflow.add_edge("html_parser", "field_detector")
        workflow.add_edge("field_detector", "question_generator")
        workflow.add_edge("question_generator", "profile_retriever")
        workflow.add_edge("profile_retriever", "ai_answerer")
        workflow.add_edge("ai_answerer", "qa_merger")  # New edge to Q&A merger
        workflow.add_edge("qa_merger", "action_generator")  # Updated edge
        workflow.add_edge("action_generator", "llm_action_generator")  # New edge to LLM action generator
        workflow.add_edge("llm_action_generator", "result_saver")  # Updated edge
        workflow.add_edge("result_saver", END)
        workflow.add_edge("error_handler", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "html_parser",
            self._check_for_errors,
            {
                "continue": "field_detector",
                "error": "error_handler"
            }
        )
        
        return workflow
    
    def _check_llm_action_for_field(self, field_name: str, field_value: str, llm_actions: List[Dict[str, Any]]) -> bool:
        """Check if LLM actions contain a click action for the specific field value"""
        for action in llm_actions:
            if action.get("type") == "click":
                selector = action.get("selector", "")
                # Check if the selector matches this field name and value
                if field_name in selector and field_value in selector:
                    return True
                # Also check for exact value match in selector
                if f"[value='{field_value}']" in selector:
                    return True
        return False

    def process_form(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any], profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process form using LangGraph workflow"""
        try:
            print(f"DEBUG: process_form - Starting with workflow_id: {workflow_id}, step_key: {step_key}")
            print(f"DEBUG: process_form - HTML length: {len(form_html)}")
            print(f"DEBUG: process_form - Profile data keys: {list(profile_data.keys()) if profile_data else 'None'}")
            print(f"DEBUG: process_form - Profile dummy data keys: {list(profile_dummy_data.keys()) if profile_dummy_data else 'None'}")

            # Set workflow_id for thread isolation
            self.set_workflow_id(workflow_id)
            
            # Create initial state
            initial_state = FormAnalysisState(
                workflow_id=workflow_id,
                step_key=step_key,
                form_html=form_html,
                profile_data=profile_data or {},
                profile_dummy_data=profile_dummy_data or {},
                parsed_form=None,
                detected_fields=[],
                field_questions=[],
                ai_answers=[],
                merged_qa_data=[],
                form_actions=[],
                llm_generated_actions=[],
                saved_step_data=None,
                dummy_data_usage=[],
                analysis_complete=False,
                error_details=None,
                messages=[]
            )
            
            # Run the workflow
            result = self.app.invoke(initial_state)
            
            print(f"DEBUG: process_form - Workflow completed")
            print(f"DEBUG: process_form - Result keys: {list(result.keys())}")
            
            # Check for errors
            if result.get("error_details"):
                return {
                    "success": False,
                    "error": result["error_details"],
                    "data": [],
                    "actions": []
                }
            
            # Return successful result with merged Q&A data
            return {
                "success": True,
                "data": result.get("merged_qa_data", []),  # 返回合并的问答数据
                "actions": result.get("llm_generated_actions", []),  # 返回LLM生成的动作
                "messages": result.get("messages", []),
                "processing_metadata": {
                    "fields_detected": len(result.get("detected_fields", [])),
                    "questions_generated": len(result.get("field_questions", [])),
                    "answers_generated": len(result.get("ai_answers", [])),
                    "actions_generated": len(result.get("llm_generated_actions", [])),
                    "workflow_id": workflow_id,
                    "step_key": step_key
                }
            }
            
        except Exception as e:
            print(f"DEBUG: process_form - Exception: {str(e)}")
            return {
                "success": False,
                "error": f"Form processing failed: {str(e)}",
                "data": [],
                "actions": []
            }

    def _check_for_errors(self, state: FormAnalysisState) -> Literal["continue", "error"]:
        """Check if there are errors in the state"""
        if state.get("error_details"):
            return "error"
        return "continue"

    def _html_parser_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Parse HTML form"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: HTML Parser - Starting")
            soup = BeautifulSoup(state["form_html"], 'html.parser')
            
            # Find form element
            form = soup.find("form")
            if not form:
                # If no form tag, use the entire HTML as form content
                form = soup

            state["parsed_form"] = {
                "html_content": str(form),  # 存储 HTML 字符串而不是 Tag 对象
                "action": form.get("action", "") if hasattr(form, 'get') else "",
                "method": form.get("method", "post") if hasattr(form, 'get') else "post"
            }

            # Store HTML content for question generation access
            self.step_analyzer._current_form_html = state["form_html"]

            print(f"[workflow_id:{workflow_id}] DEBUG: HTML Parser - Completed successfully")
            
        except Exception as e:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: HTML Parser - Error: {str(e)}")
            state["error_details"] = f"HTML parsing failed: {str(e)}"
        
        return state

    def _field_detector_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Detect form fields with optimization"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Field Detector - Starting with optimizations")
            
            if not state["parsed_form"]:
                state["error_details"] = "No parsed form available"
                return state
            
            # 修复：从 HTML 字符串重新解析 BeautifulSoup 对象
            html_content = state["parsed_form"]["html_content"]
            form_elements = BeautifulSoup(html_content, 'html.parser')
            
            detected_fields = []
            processed_field_groups = set()  # Track processed radio/checkbox groups
            field_groups = {}  # Track field groups for optimization
            
            # Find all form input elements
            if hasattr(form_elements, 'find_all'):
                for element in form_elements.find_all(["input", "select", "textarea"]):
                    element_type = element.get("type", "text").lower() if element.name == "input" else element.name.lower()
                    element_name = element.get("name", "")
                    
                    # For radio and checkbox fields, only process each group once
                    if element_type in ["radio", "checkbox"] and element_name:
                        group_key = f"{element_type}_{element_name}"
                        if group_key in processed_field_groups:
                            print(f"DEBUG: Field Detector - Skipping duplicate {element_type} field: {element_name}")
                            continue
                        processed_field_groups.add(group_key)
                        print(f"DEBUG: Field Detector - Processing {element_type} group: {element_name}")

                        # 🚀 OPTIMIZATION: Group related radio/checkbox fields
                        base_name = re.sub(r'\[\d+\]$', '', element_name)
                        if base_name not in field_groups:
                            field_groups[base_name] = []
                    
                    field_info = self.step_analyzer._extract_field_info(element)
                    if field_info:
                        detected_fields.append(field_info)

                        # Add to field group if it's a radio/checkbox
                        if element_type in ["radio", "checkbox"] and element_name:
                            base_name = re.sub(r'\[\d+\]$', '', element_name)
                            if base_name in field_groups:
                                field_groups[base_name].append(field_info)
                        
                        print(f"DEBUG: Field Detector - Found field: {field_info['name']} ({field_info['type']})")
            
            state["detected_fields"] = detected_fields
            state["field_groups_info"] = field_groups  # Store grouping info
            
            print(f"DEBUG: Field Detector - Found {len(detected_fields)} fields (after deduplication)")
            print(f"DEBUG: Field Detector - Created {len(field_groups)} field groups")
            
        except Exception as e:
            print(f"DEBUG: Field Detector - Error: {str(e)}")
            state["error_details"] = f"Field detection failed: {str(e)}"
        
        return state

    def _question_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate questions for form fields with grouping optimization"""
        try:
            print("DEBUG: Question Generator - Starting with optimizations")
            
            # Extract page context for better question generation
            page_context = self.step_analyzer._extract_page_context(state["parsed_form"])
            self.step_analyzer._page_context = page_context

            # 🚀 OPTIMIZATION: Use field groups to generate better questions
            field_groups_info = state.get("field_groups_info", {})
            detected_fields = state["detected_fields"]
            
            questions = []
            processed_fields = set()

            # Process grouped fields first
            for base_name, group_fields in field_groups_info.items():
                if len(group_fields) > 1:
                    # Generate grouped question for radio/checkbox groups
                    grouped_question = self._generate_grouped_question(group_fields, base_name)
                    questions.append(grouped_question)

                    # Mark fields as processed
                    for field in group_fields:
                        processed_fields.add(field.get("id", ""))

                    print(
                        f"DEBUG: Question Generator - Generated grouped question for {base_name}: {grouped_question['question']}")

            # Process remaining individual fields
            for field in detected_fields:
                if field.get("id", "") not in processed_fields:
                    question = self.step_analyzer._generate_field_question(field)
                    questions.append(question)
                    print(
                        f"DEBUG: Question Generator - Generated individual question for {field['name']}: {question['question']}")
            
            state["field_questions"] = questions
            print(
                f"DEBUG: Question Generator - Generated {len(questions)} questions ({len(field_groups_info)} grouped)")
            
        except Exception as e:
            print(f"DEBUG: Question Generator - Error: {str(e)}")
            state["error_details"] = f"Question generation failed: {str(e)}"
        
        return state

    def _profile_retriever_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Retrieve user profile data (already provided in state)"""
        try:
            print("DEBUG: Profile Retriever - Starting")
            
            # Profile data is already in state, just validate it
            profile_data = state.get("profile_data", {})
            
            if not profile_data:
                print("DEBUG: Profile Retriever - No profile data provided")
                state["messages"].append({
                    "type": "warning",
                    "content": "No profile data provided - answers may be limited"
                })
            else:
                print(f"DEBUG: Profile Retriever - Profile data available with {len(profile_data)} keys")
            
        except Exception as e:
            print(f"DEBUG: Profile Retriever - Error: {str(e)}")
            state["error_details"] = f"Profile retrieval failed: {str(e)}"
        
        return state

    def _ai_answerer_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate AI answers for questions"""
        try:
            print("DEBUG: AI Answerer - Starting")
            
            answers = []
            dummy_usage = []
            profile_data = state.get("profile_data", {})
            profile_dummy_data = state.get("profile_dummy_data", {})
            
            for question in state["field_questions"]:
                answer = self.step_analyzer._generate_ai_answer(question, profile_data, profile_dummy_data)
                answers.append(answer)
                
                # Track dummy data usage (including AI-generated dummy data)
                if answer.get("used_dummy_data", False):
                    dummy_source = answer.get("dummy_data_source", "unknown")
                    dummy_usage.append({
                        "question": question.get("question", ""),
                        "field_name": question.get("field_name", ""),
                        "answer": answer.get("answer", ""),
                        "dummy_data_source": dummy_source,
                        "dummy_data_type": answer.get("dummy_data_type", "unknown"),
                        "confidence": answer.get("confidence", 0),
                        "reasoning": answer.get("reasoning", "")
                    })
                    
                    if dummy_source == "ai_generated":
                        print(f"DEBUG: AI Answerer - Generated smart dummy data for {question['field_name']}: {answer['answer']} (confidence: {answer['confidence']})")
                    else:
                        print(f"DEBUG: AI Answerer - Used provided dummy data for {question['field_name']}: {answer['answer']}")
                else:
                    print(f"DEBUG: AI Answerer - Used profile data for {question['field_name']}: confidence={answer['confidence']}")
            
            state["ai_answers"] = answers
            state["dummy_data_usage"] = dummy_usage
            print(f"DEBUG: AI Answerer - Generated {len(answers)} answers, {len(dummy_usage)} used dummy data")
            
        except Exception as e:
            print(f"DEBUG: AI Answerer - Error: {str(e)}")
            state["error_details"] = f"AI answer generation failed: {str(e)}"
        
        return state

    def _qa_merger_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Merge questions and answers into unified data structure"""
        try:
            print("DEBUG: Q&A Merger - Starting")
            
            questions = state.get("field_questions", [])
            answers = state.get("ai_answers", [])

            print(f"DEBUG: Q&A Merger - Processing {len(questions)} questions and {len(answers)} answers")

            # Validate that we have questions to process
            if not questions:
                print("DEBUG: Q&A Merger - No questions to process")
                state["merged_qa_data"] = []
                return state
            
            # Group questions by question text to merge related fields
            question_groups = {}
            for i, question in enumerate(questions):
                # Validate question structure
                if not isinstance(question, dict):
                    print(f"DEBUG: Q&A Merger - Skipping invalid question at index {i}: not a dict")
                    continue

                if "question" not in question:
                    print(f"DEBUG: Q&A Merger - Skipping question at index {i}: missing 'question' field")
                    continue
                
                question_text = question["question"]
                if question_text not in question_groups:
                    question_groups[question_text] = []
                question_groups[question_text].append(question)

            print(f"DEBUG: Q&A Merger - Created {len(question_groups)} question groups")
            
            merged_data = []
            
            for question_text, grouped_questions in question_groups.items():
                print(f"DEBUG: Q&A Merger - Processing question group: '{question_text}' with {len(grouped_questions)} fields")

                # Validate grouped questions
                valid_questions = []
                for question in grouped_questions:
                    required_fields = ["field_selector", "field_name", "field_type", "field_label", "required", "id"]
                    missing_fields = [field for field in required_fields if field not in question]

                    if missing_fields:
                        print(f"DEBUG: Q&A Merger - Question missing fields {missing_fields}: {question}")
                        # Try to provide default values for missing fields
                        for field in missing_fields:
                            if field == "field_selector":
                                question[field] = f"[data-field='{question.get('field_name', 'unknown')}']"
                            elif field == "field_name":
                                question[field] = f"field_{len(valid_questions)}"
                            elif field == "field_type":
                                question[field] = "text"
                            elif field == "field_label":
                                question[field] = question.get("question", "Unknown Field")
                            elif field == "required":
                                question[field] = False
                            elif field == "id":
                                question[field] = f"q_{question.get('field_name', 'unknown')}_{uuid.uuid4().hex[:8]}"

                        print(f"DEBUG: Q&A Merger - Added default values for missing fields")

                    valid_questions.append(question)

                if not valid_questions:
                    print(f"DEBUG: Q&A Merger - No valid questions in group '{question_text}', skipping")
                    continue
                
                # Determine the primary question (usually the first one or the main input field)
                primary_question = self._find_primary_question(valid_questions)
                
                # Collect all related fields and their answers
                all_field_data = []
                all_needs_intervention = []
                all_confidences = []
                all_reasonings = []

                for question in valid_questions:
                    # Find corresponding answer
                    ai_answer = self.step_analyzer._find_answer_for_question(question, answers)
                    
                    # Determine if intervention is needed for this field
                    needs_intervention = (not ai_answer or 
                                        ai_answer.get("needs_intervention", False) or 
                                        ai_answer.get("confidence", 0) < 50)
                    
                    all_needs_intervention.append(needs_intervention)
                    all_confidences.append(ai_answer.get("confidence", 0) if ai_answer else 0)
                    all_reasonings.append(ai_answer.get("reasoning", "") if ai_answer else "")
                    
                    # Create answer data for this field
                    field_answer_data = self.step_analyzer._create_answer_data(question, ai_answer, needs_intervention)
                    all_field_data.extend(field_answer_data)
                
                # Determine the overall answer type based on the field types in the group
                answer_type = self._determine_group_answer_type(valid_questions)
                
                # Determine overall intervention status (if any field needs intervention)
                overall_needs_intervention = any(all_needs_intervention)
                
                # CRITICAL: Check if any answer data has check=1 (indicating an answer exists)
                # If any answer has check=1, then no interrupt should be set regardless of needs_intervention
                has_valid_answer = any(item.get("check", 0) == 1 for item in all_field_data)
                
                # Interrupt logic: only set interrupt if needs intervention AND no valid answer exists
                should_interrupt = overall_needs_intervention and not has_valid_answer
                
                print(f"DEBUG: Q&A Merger - Question '{question_text}': needs_intervention={overall_needs_intervention}, has_valid_answer={has_valid_answer}, should_interrupt={should_interrupt}")
                
                # Calculate average confidence
                avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                
                # Combine all reasoning
                combined_reasoning = "; ".join(filter(None, all_reasonings))

                # Extract the correct selector from the first valid field data
                correct_selector = primary_question.get("field_selector", "")
                if all_field_data:
                    # Use the selector from the first field data if available (this will be the corrected one)
                    first_field_selector = all_field_data[0].get("selector", "")
                    if first_field_selector:
                        correct_selector = first_field_selector
                
                # Create merged data structure
                merged_item = {
                    "question": {
                        "data": {
                            "name": question_text  # Always use question text for question.data.name
                        },
                        "answer": {
                            "type": answer_type,
                            "selector": correct_selector,  # Use the corrected selector
                            "data": all_field_data  # Combined data from all related fields
                        }
                    },
                    # Keep metadata for internal use (using primary question's metadata)
                    "_metadata": {
                        "id": primary_question.get("id", ""),
                        "field_selector": primary_question.get("field_selector", ""),
                        "field_name": primary_question.get("field_name", ""),
                        "field_type": primary_question.get("field_type", ""),
                        "field_label": primary_question.get("field_label", ""),
                        "required": primary_question.get("required", False),
                        "options": self._combine_all_options(valid_questions),  # Combined options from all fields
                        "confidence": avg_confidence,
                        "reasoning": combined_reasoning,
                        "needs_intervention": overall_needs_intervention,
                        "has_valid_answer": has_valid_answer,  # Track if answer exists
                        "grouped_fields": [q.get("field_name", "") for q in valid_questions]  # Track all grouped fields
                    }
                }
                
                # Add interrupt field at question level ONLY if should_interrupt is True
                if should_interrupt:
                    merged_item["question"]["type"] = "interrupt"
                    interrupt_status = "interrupt"
                else:
                    interrupt_status = "normal" if has_valid_answer else "no_answer_but_no_interrupt"
                
                merged_data.append(merged_item)
                print(
                    f"DEBUG: Q&A Merger - Merged question '{question_text}': type={answer_type}, fields={len(valid_questions)}, status={interrupt_status}")
            
            state["merged_qa_data"] = merged_data

            # 🚀 OPTIMIZATION: Consistency checking
            consistency_issues = self._check_answer_consistency(merged_data)
            state["consistency_issues"] = consistency_issues

            if consistency_issues:
                print(f"DEBUG: Q&A Merger - Found {len(consistency_issues)} consistency issues")
                for issue in consistency_issues:
                    print(f"DEBUG: Consistency Issue - {issue['type']}: {issue['message']}")

                # Mark critical issues for intervention
                critical_issues = [i for i in consistency_issues if
                                   i['type'] in ['radio_selection_error', 'conflicting_values']]
                if critical_issues:
                    print("DEBUG: Critical consistency issues found, marking for intervention")
                    for item in merged_data:
                        for issue in critical_issues:
                            if issue.get('field_name') == item.get('_metadata', {}).get('field_name'):
                                item['question']['type'] = 'interrupt'
            else:
                print("DEBUG: Q&A Merger - No consistency issues found")
            
            print(f"DEBUG: Q&A Merger - Created {len(merged_data)} merged question groups from {len(questions)} original questions")
            
        except Exception as e:
            print(f"DEBUG: Q&A Merger - Error: {str(e)}")
            import traceback
            traceback.print_exc()
            state["error_details"] = f"Q&A merging failed: {str(e)}"
        
        return state

    def _find_primary_question(self, grouped_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the primary question from a group of related questions"""
        # Priority order: input fields first, then radio, then checkbox, then select
        type_priority = {"input": 1, "number": 1, "text": 1, "email": 1, "tel": 1, 
                        "radio": 2, "checkbox": 3, "select": 4}
        
        # Sort by type priority, then by field name length (shorter names are often primary)
        sorted_questions = sorted(grouped_questions, 
                                key=lambda q: (type_priority.get(q["field_type"], 5), len(q["field_name"])))
        
        return sorted_questions[0]

    def _determine_group_answer_type(self, grouped_questions: List[Dict[str, Any]]) -> str:
        """Determine the answer type for a group of related questions based on HTML field types"""
        if not grouped_questions:
            return "input"
        
        # Get the primary question (most representative field)
        primary_question = self._find_primary_question(grouped_questions)
        primary_field_type = primary_question["field_type"]
        primary_field_name = primary_question.get("field_name", "unknown")
        
        print(f"DEBUG: _determine_group_answer_type - Group has {len(grouped_questions)} fields")
        print(f"DEBUG: _determine_group_answer_type - Primary field: '{primary_field_name}' (type: {primary_field_type})")
        
        # Directly map the primary field's HTML type to answer component type
        answer_type = self._map_field_type_to_answer_type(primary_field_type)
        
        print(f"DEBUG: _determine_group_answer_type - Final answer type: '{answer_type}'")
        
        return answer_type

    def _combine_all_options(self, grouped_questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Combine options from all questions in a group"""
        all_options = []
        seen_values = set()
        
        for question in grouped_questions:
            for option in question.get("options", []):
                option_value = option.get("value", "")
                if option_value and option_value not in seen_values:
                    all_options.append(option)
                    seen_values.add(option_value)
        
        return all_options

    def _map_field_type_to_answer_type(self, field_type: str) -> str:
        """Map HTML field type to answer component type"""
        type_mapping = {
            # Text input types
            "text": "input",
            "email": "input", 
            "password": "input",
            "number": "input",
            "tel": "input",
            "url": "input",
            "search": "input",
            "date": "input",
            "time": "input",
            "datetime-local": "input",
            "month": "input",
            "week": "input",
            "color": "input",
            "range": "input",
            "file": "input",
            "textarea": "input",
            
            # Selection types
            "radio": "radio",
            "checkbox": "checkbox", 
            "select": "select",
            "select-one": "select",
            "select-multiple": "select",
            
            # Button types (usually not for data input)
            "button": "button",
            "submit": "button",
            "reset": "button",
            "image": "button",
            
            # Hidden and other types
            "hidden": "input"
        }
        
        field_type_lower = field_type.lower()
        answer_type = type_mapping.get(field_type_lower, "input")
        
        print(f"DEBUG: _map_field_type_to_answer_type - HTML field type '{field_type}' mapped to answer type '{answer_type}'")
        
        return answer_type

    def _generate_grouped_question(self, group_fields: List[Dict[str, Any]], base_name: str) -> Dict[str, Any]:
        """🚀 OPTIMIZATION: Generate comprehensive question for field group"""
        try:
            if not group_fields:
                return {}

            # Find primary field (usually first or most representative)
            primary_field = group_fields[0]
            for field in group_fields:
                if field.get("type") in ["input", "text"]:
                    primary_field = field
                    break

            # Collect all options from the group
            all_options = []
            for field in group_fields:
                for option in field.get("options", []):
                    if option not in all_options:
                        all_options.append(option)

            # Generate comprehensive question based on field type
            field_type = primary_field.get("type", "")

            if "telephone" in base_name.lower() or "phone" in base_name.lower():
                question_text = "What are your telephone number preferences and details?"
            elif "email" in base_name.lower():
                question_text = "Do you have another email address?"
            elif field_type in ["radio", "checkbox"]:
                # Create question that shows all available options
                option_texts = [opt.get("text", opt.get("value", "")) for opt in all_options]
                if len(option_texts) <= 3:
                    options_str = ", ".join(option_texts)
                    question_text = f"Please select from the following options: {options_str}"
                else:
                    question_text = f"Please make your selection from the available options"
            else:
                field_label = primary_field.get("label", base_name)
                question_text = f"Please provide information for: {field_label}"

            # Create grouped question structure
            grouped_question = {
                "id": f"group_{base_name}_{uuid.uuid4().hex[:8]}",
                "field_selector": primary_field.get("selector", ""),
                "field_name": base_name,
                "field_type": field_type,
                "field_label": primary_field.get("label", base_name),
                "question": question_text,
                "required": any(f.get("required", False) for f in group_fields),
                "options": all_options,
                "grouped_fields": group_fields,  # Keep reference to all fields in group
                "is_grouped": True,
                "group_size": len(group_fields)
            }

            print(
                f"DEBUG: Generated grouped question for {base_name} with {len(group_fields)} fields and {len(all_options)} options")

            return grouped_question

        except Exception as e:
            print(f"DEBUG: Error generating grouped question for {base_name}: {str(e)}")
            # Fallback to first field's question
            if group_fields:
                return self.step_analyzer._generate_field_question(group_fields[0])
            return {}

    def _check_answer_consistency(self, merged_qa_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """🚀 OPTIMIZATION: Check for logical inconsistencies in answers"""
        consistency_issues = []

        try:
            for answer_data in merged_qa_data:
                metadata = answer_data.get("_metadata", {})
                field_type = metadata.get("field_type", "")
                field_name = metadata.get("field_name", "")

                # Check radio/checkbox group consistency
                if field_type in ["radio", "checkbox"]:
                    issues = self._check_radio_checkbox_consistency(answer_data)
                    consistency_issues.extend(issues)

                # Check input group consistency
                elif field_type in ["text", "email", "tel", "number"]:
                    issues = self._check_input_group_consistency(answer_data)
                    consistency_issues.extend(issues)

                # Check for empty required fields
                if metadata.get("required", False):
                    question_data = answer_data.get("question", {})
                    answer_info = question_data.get("answer", {})
                    data_array = answer_info.get("data", [])

                    has_answer = any(item.get("check") == 1 for item in data_array)
                    if not has_answer:
                        consistency_issues.append({
                            "type": "required_field_empty",
                            "message": f"Required field '{field_name}' has no answer",
                            "field_name": field_name
                        })

        except Exception as e:
            print(f"DEBUG: Consistency check error: {str(e)}")
            consistency_issues.append({
                "type": "consistency_check_error",
                "message": f"Error during consistency check: {str(e)}",
                "field_name": "unknown"
            })

        return consistency_issues

    def _check_radio_checkbox_consistency(self, answer_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check consistency for radio/checkbox groups"""
        issues = []

        try:
            question_data = answer_data.get("question", {})
            answer_info = question_data.get("answer", {})
            data_array = answer_info.get("data", [])
            metadata = answer_data.get("_metadata", {})
            field_type = metadata.get("field_type", "")
            field_name = metadata.get("field_name", "")

            # Count selected items
            selected_count = sum(1 for item in data_array if item.get("check") == 1)
            selected_values = [item.get("value", "") for item in data_array if item.get("check") == 1]

            # Radio buttons should have exactly one selection
            if field_type == "radio" and selected_count > 1:
                issues.append({
                    "type": "radio_selection_error",
                    "message": f"Radio group should have exactly 1 selection, found {selected_count}",
                    "field_name": field_name
                })

            # Check for conflicting boolean values
            if "true" in selected_values and "false" in selected_values:
                issues.append({
                    "type": "conflicting_values",
                    "message": "Cannot select both 'true' and 'false' values",
                    "field_name": field_name
                })

            # Check for conflicting yes/no values
            yes_values = ["yes", "true", "1"]
            no_values = ["no", "false", "0"]

            has_yes = any(val.lower() in yes_values for val in selected_values)
            has_no = any(val.lower() in no_values for val in selected_values)

            if has_yes and has_no:
                issues.append({
                    "type": "conflicting_yes_no",
                    "message": "Cannot select both yes and no values",
                    "field_name": field_name
                })

        except Exception as e:
            issues.append({
                "type": "radio_checkbox_check_error",
                "message": f"Error checking radio/checkbox consistency: {str(e)}",
                "field_name": answer_data.get("_metadata", {}).get("field_name", "unknown")
            })

        return issues

    def _check_input_group_consistency(self, answer_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check consistency for input field groups"""
        issues = []

        try:
            question_data = answer_data.get("question", {})
            answer_info = question_data.get("answer", {})
            data_array = answer_info.get("data", [])
            metadata = answer_data.get("_metadata", {})
            field_name = metadata.get("field_name", "")

            # Get filled values
            filled_values = [item.get("value", "") for item in data_array if
                             item.get("check") == 1 and item.get("value")]

            # Check email format consistency
            if "email" in field_name.lower():
                for value in filled_values:
                    if value and "@" not in value:
                        issues.append({
                            "type": "invalid_email_format",
                            "message": f"Invalid email format: {value}",
                            "field_name": field_name
                        })

            # Check phone number format consistency
            if "telephone" in field_name.lower() or "phone" in field_name.lower():
                for value in filled_values:
                    if value and not re.match(r'^[\d\s\-\+\(\)]+$', value):
                        issues.append({
                            "type": "invalid_phone_format",
                            "message": f"Invalid phone format: {value}",
                            "field_name": field_name
                        })

        except Exception as e:
            issues.append({
                "type": "input_group_check_error",
                "message": f"Error checking input group consistency: {str(e)}",
                "field_name": answer_data.get("_metadata", {}).get("field_name", "unknown")
            })

        return issues

    def _action_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate form actions from merged Q&A data"""
        try:
            print("DEBUG: Action Generator - Starting")
            
            actions = []
            merged_data = state.get("merged_qa_data", [])
            
            for item in merged_data:
                # Extract metadata for action generation
                metadata = item.get("_metadata", {})
                question_data = item.get("question", {})

                # CHANGED: Don't skip interrupt fields - generate basic actions for them too
                is_interrupt = question_data.get("type") == "interrupt"
                print(
                    f"DEBUG: Action Generator - Processing {metadata.get('field_name', 'unknown')} (interrupt: {is_interrupt})")
                
                answer_data = question_data.get("answer", {})

                # Extract answer value from the data array
                answer_value = self._extract_answer_from_data(answer_data)

                print(f"DEBUG: Action Generator - Processing {metadata.get('field_name', 'unknown')}")
                print(f"DEBUG: Action Generator - Field type: {metadata.get('field_type', 'unknown')}")
                print(f"DEBUG: Action Generator - Answer value: '{answer_value}'")
                print(f"DEBUG: Action Generator - Needs intervention: {metadata.get('needs_intervention', False)}")
                print(f"DEBUG: Action Generator - Has valid answer: {metadata.get('has_valid_answer', False)}")

                # CRITICAL: Always generate action with selector and type, even if no answer
                print(
                    f"DEBUG: Action Generator - Processing field {metadata.get('field_name', 'unknown')} - answer: '{answer_value}'")

                # For checkbox fields, we need to generate actions for each checked item
                if metadata.get("field_type") == "checkbox":
                    data_array = answer_data.get("data", [])
                    checkbox_actions_generated = False
                    for data_item in data_array:
                        if data_item.get("check") == 1:
                            # Generate action for this specific checked item
                            action = {
                                "selector": data_item.get("selector", metadata.get("field_selector", "")),
                                "type": "click",
                                "value": data_item.get("value", "")
                            }
                            actions.append(action)
                            checkbox_actions_generated = True
                            print(f"DEBUG: Action Generator - Generated checkbox action: {action}")

                    if checkbox_actions_generated:
                        continue  # Skip the traditional action generation only if we generated checkbox actions

                # For non-checkbox fields or checkboxes without checked items, use traditional generation
                # Generate action for input fields and other field types
                if metadata.get("field_type") in ["text", "email", "password", "number", "tel", "url", "date", "time",
                                                  "datetime-local", "textarea", "select"]:
                    data_array = answer_data.get("data", [])
                    for data_item in data_array:
                        if data_item.get("check") == 1:
                            # Generate input action for this field
                            action = {
                                "selector": data_item.get("selector", metadata.get("field_selector", "")),
                                "type": "input",
                                "value": data_item.get("value", "")
                            }
                            actions.append(action)
                            print(f"DEBUG: Action Generator - Generated input action: {action}")
                            break  # Only need one input action per field
                    continue  # Skip traditional generation for input fields
                
                # Create a compatible data structure for the existing action generator
                compatible_item = {
                    "field_selector": metadata.get("field_selector", ""),
                    "field_name": metadata.get("field_name", ""),
                    "field_type": metadata.get("field_type", ""),
                    "options": metadata.get("options", []),
                    "confidence": metadata.get("confidence", 0),
                    "reasoning": metadata.get("reasoning", ""),
                    "answer": answer_value
                }

                # Try to generate action using traditional method
                try:
                    action_result = self.step_analyzer._generate_form_action(compatible_item)
                    if action_result:
                        if isinstance(action_result, list):
                            # Handle multiple actions (e.g., multiple checkboxes)
                            for action in action_result:
                                print(
                                    f"DEBUG: Action Generator - Generated action for {metadata.get('field_name', 'unknown')}: {action['type']} = '{action.get('value', 'N/A')}'")
                            actions.extend(action_result)
                            # Continue to process next field - don't break here
                        else:
                            # Handle single action (legacy format)
                            print(
                                f"DEBUG: Action Generator - Generated action for {metadata.get('field_name', 'unknown')}: {action_result['type']} = '{action_result.get('value', 'N/A')}'")
                            actions.append(action_result)
                            # Continue to process next field - don't break here
                except Exception as action_error:
                    print(
                        f"DEBUG: Action Generator - Failed to generate action for {metadata.get('field_name', 'unknown')}: {str(action_error)}")
                    continue

            # Sort actions by order (not needed for new format, but keep for compatibility)
            # actions.sort(key=lambda x: x.get("order", 0))
            
            state["form_actions"] = actions
            print(f"DEBUG: Action Generator - Generated {len(actions)} actions total")

            # Debug: Print all generated actions
            for i, action in enumerate(actions, 1):
                print(
                    f"DEBUG: Action {i}: {action.get('selector', 'no selector')} -> {action.get('type', 'no type')} ({action.get('value', 'no value')})")
            
        except Exception as e:
            print(f"DEBUG: Action Generator - Error: {str(e)}")
            import traceback
            traceback.print_exc()
            state["error_details"] = f"Action generation failed: {str(e)}"
        
        return state

    def _extract_answer_from_data(self, answer_data: Dict[str, Any]) -> str:
        """Extract answer value from answer data structure"""
        data_array = answer_data.get("data", [])

        # Find all checked items
        checked_items = []
        for item in data_array:
            if item.get("check") == 1:
                value = item.get("value", item.get("name", ""))
                # Ensure value is a string
                if isinstance(value, list):
                    # If value is a list, join it or take first element
                    value = ",".join(str(v) for v in value) if value else ""
                elif value is None:
                    value = ""
                else:
                    value = str(value)

                if value:  # Only add non-empty values
                    checked_items.append(value)

        # Return comma-separated values for multiple selections
        if checked_items:
            return ",".join(checked_items)
        
        # If no checked item, return empty string
        return ""

    def _llm_action_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Use LLM to generate actions in the format required by graph.py"""
        try:
            print("DEBUG: LLM Action Generator - Starting")

            # Collect dummy data that was generated in previous steps
            dummy_data_context = {}
            merged_data = state.get("merged_qa_data", [])

            # Check if we have any form fields to process
            if not merged_data:
                print(
                    "DEBUG: LLM Action Generator - No form fields detected, checking for task list or direct HTML processing")
                # This might be a task list page or other non-form page
                # Let LLM directly analyze the HTML for clickable elements
                return self._process_non_form_page(state)

            # Extract dummy data from merged_qa_data for LLM context
            for item in merged_data:
                metadata = item.get("_metadata", {})
                question_data = item.get("question", {})
                answer_data = question_data.get("answer", {})

                field_name = metadata.get("field_name", "")
                field_selector = metadata.get("field_selector", "")

                # Extract any existing answers (including dummy data)
                data_array = answer_data.get("data", [])
                for data_item in data_array:
                    if data_item.get("check") == 1:
                        dummy_data_context[field_name] = {
                            "selector": field_selector,
                            "value": data_item.get("value", data_item.get("name", "")),
                            "field_type": metadata.get("field_type", ""),
                            "reasoning": metadata.get("reasoning", "")
                        }
                        break

            print(f"DEBUG: LLM Action Generator - Extracted dummy data context: {dummy_data_context}")

            # Prepare enhanced prompt with HTML, profile data, and dummy data context
            prompt = f"""
            # Role 
            You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
            analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
            filling in the form content, and returning it to the front end in a fixed json format. only json information is needed

            # HTML Content:
            {state["form_html"]}

            # User Profile Data:
            {json.dumps(state["profile_data"], indent=2, ensure_ascii=False)}

            # Previously Generated Dummy Data (USE THESE VALUES):
            {json.dumps(dummy_data_context, indent=2, ensure_ascii=False)}

            # CRITICAL INSTRUCTIONS FOR DATA USAGE:
            ## Data Priority Rules (MUST FOLLOW):
            1. **HIGHEST PRIORITY**: Use the "Previously Generated Dummy Data" above for fields that have been analyzed
            2. **SECOND PRIORITY**: Use REAL data from User Profile Data for any remaining fields
            3. **EXACT matching priority**: Look for exact field name matches first (e.g., "telephoneNumber" -> telephoneNumber)
            4. **Semantic matching**: If no exact match, find semantically similar fields (e.g., "phone" -> telephoneNumber)
            5. **Nested data access**: User profile data may be nested (e.g., contactInformation.telephoneNumber)
            6. **Data type conversion**: Convert data types as needed (e.g., boolean to checkbox selection)
            7. **Required field priority**: Always fill required fields first
            8. **If no matching data exists**: Generate reasonable dummy data for that field

            ## Data Matching Examples:
            - HTML field "telephoneNumber" -> Previously Generated Dummy Data "telephoneNumber": "5551234567"
            - HTML field "telephoneNumberPurpose" -> Previously Generated Dummy Data "telephoneNumberPurpose[0]": "useInUK"
            - HTML field "email" -> User data "contactInformation.primaryEmail": "user@email.com"
            - HTML field "firstName" -> User data "personalDetails.firstName": "John"

            # Response json format (EXACTLY like this):
            {{
              "actions": [
                {{
                  "selector": "input[type='number'][name='telephoneNumber']",
                  "type": "input",
                  "value": "5551234567"
                }},
                {{
                  "selector": "input[type='checkbox'][name='telephoneNumberPurpose[0]'][value='useInUK']",
                  "type": "click"
                }},
                {{
                  "selector": "input[type='checkbox'][name='telephoneNumberType[2]'][value='mobile']",
                  "type": "click"
                }},
                {{
                  "selector": "input[type='submit']",
                  "type": "click"
                }}
              ]
            }}

            # Instructions:
            ## For Traditional Forms:
            1. **FIRST**: Check the "Previously Generated Dummy Data" section for any pre-analyzed fields
            2. **SECOND**: Analyze the User Profile Data structure to understand available real data
            3. **THIRD**: Analyze the HTML form elements to identify fillable fields
            4. **FOURTH**: Match form fields with appropriate data using the priority rules above
            5. **FIFTH**: Generate CSS selectors for each form element that needs to be filled
            6. For input fields (text, email, password, number, etc.): use "type": "input" and provide "value" from dummy data, profile data, or reasonable new dummy data
            7. For radio buttons and checkboxes: use "type": "click" (no value needed) - select based on dummy data, profile data, or reasonable choices
            8. For select dropdowns: use "type": "input" and provide "value" from dummy data or profile data
            9. **IMPORTANT**: For checkbox groups that appear to be mutually exclusive (like "purpose" or "type"), select only ONE option
            10. **ALWAYS add submit action**: After filling form fields, add a submit button click action to proceed to next step

            ## For Task List Pages (Government/Application Pages):
            11. Look for task lists with status indicators (like "In progress", "Cannot start yet", "Completed")
            12. For tasks with "In progress" status that have clickable links: generate "type": "click" actions
            13. For tasks with "Cannot start yet" status: do NOT generate actions (skip them)
            14. For completed tasks: generally skip unless specifically needed
            15. Use the exact href attribute value in the selector: a[href="exact-path"]

            ## Priority Rules:
            - **HIGHEST PRIORITY**: Use previously generated dummy data values for consistency
            - **SECOND PRIORITY**: Generate actions for all fillable form fields with real or reasonable dummy data
            - If the page contains a task list with "In progress" items, prioritize clicking those links
            - If the page is a traditional form, prioritize filling form fields with dummy data first, then profile data, then add submit action
            - If both exist, handle both types of actions
            - Always use precise CSS selectors that will uniquely identify the element
            - **ALWAYS end with submit/continue button action for forms**

            ## CSS Selector Examples:
            - For links: a[href="/STANDALONE_ACCOUNT_REGISTRATION/3434-0940-8939-9272/identity"]
            - For form inputs: input[type="text"][name="firstName"]
            - For number inputs: input[type="number"][name="telephoneNumber"]
            - For radio buttons: input[type="radio"][name="gender"][value="male"]
            - For checkboxes: input[type="checkbox"][name="interests"][value="sports"]
            - For select dropdowns: select[name="country"]
            - For submit buttons: input[type="submit"] or button[type="submit"]

            # Analysis Steps:
            1. **Check Previously Generated Dummy Data**: Use these values first for any matching fields
            2. **Parse User Profile Data**: Understand the structure and available real data
            3. **Determine page type**: Task list page or traditional form
            4. **If task list**: Identify tasks with "In progress" status and clickable links
            5. **If traditional form**: 
               a. Identify all fillable form fields (input, select, radio, checkbox)
               b. Match each field with dummy data first, then profile data using priority rules
               c. Generate actions with the matched data values
               d. For mutually exclusive checkboxes (purpose/type/category), select only ONE option
               e. Generate one action per form field that needs to be filled
               f. Add submit button action at the end
            6. **Generate appropriate actions** based on the page type and available data
            7. **Return ONLY the JSON response**, no other text

            ## REMEMBER:
            - **FIRST**: Use Previously Generated Dummy Data for consistency
            - **SECOND**: Use REAL data from the User Profile Data when available
            - Generate reasonable dummy data if no real data matches (e.g., for telephone: use format like "555-123-4567")
            - For checkbox groups with "purpose", "type", "category" in the name, select ONLY ONE option (they are usually mutually exclusive)
            - Include the actual data values in the "value" field for input actions
            - Always end with a submit button click to proceed to the next step
            - Pay attention to nested data structures in both dummy data and profile data

            Please provide the complete JSON response for this page:
            """

            print(f"DEBUG: LLM Action Generator - Sending enhanced prompt to LLM with dummy data context")
            print(f"DEBUG: LLM Action Generator - Profile data keys: {list(state['profile_data'].keys())}")
            print(f"DEBUG: LLM Action Generator - Dummy data fields: {list(dummy_data_context.keys())}")

            # Use LLM call with workflow_id for logging
            if self._current_workflow_id:
                response = self._invoke_llm([HumanMessage(content=prompt)], self._current_workflow_id)
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
            
            print(f"DEBUG: LLM Action Generator - Received response: {response.content}")

            try:
                # Try to parse JSON response using robust parsing
                llm_result = robust_json_parse(response.content)
                
                if "actions" in llm_result:
                    actions = llm_result["actions"]

                    # Check if submit button action exists
                    has_submit = any(
                        "submit" in action.get("selector", "").lower() or
                        action.get("type") == "submit" or
                        ("button" in action.get("selector", "").lower() and "submit" in action.get("selector",
                                                                                                   "").lower())
                        for action in actions
                    )

                    # If no submit action found, automatically add one by finding submit button in HTML
                    if not has_submit:
                        print(
                            "DEBUG: LLM Action Generator - No submit action found, searching for submit button in HTML")
                        submit_action = self._find_and_create_submit_action(state["form_html"])
                        if submit_action:
                            actions.append(submit_action)
                            print(f"DEBUG: LLM Action Generator - Added submit action: {submit_action}")
                        else:
                            print("DEBUG: LLM Action Generator - No submit button found in HTML")
                    
                    # Store the LLM-generated actions in the dedicated field
                    state["llm_generated_actions"] = actions
                    print(f"DEBUG: LLM Action Generator - Successfully parsed {len(actions)} actions")
                    
                    # Log the types of actions generated for debugging
                    action_types = {}
                    action_values = []
                    for action in actions:
                        action_type = action.get("type", "unknown")
                        action_types[action_type] = action_types.get(action_type, 0) + 1
                        if action.get("value"):
                            action_values.append(f"{action.get('selector', 'unknown')}: {action.get('value')}")
                    print(f"DEBUG: LLM Action Generator - Action types: {action_types}")
                    print(f"DEBUG: LLM Action Generator - Action values: {action_values}")
                    
                    state["messages"].append({
                        "type": "system",
                        "content": f"LLM generated {len(actions)} actions successfully using dummy data context. Types: {action_types}"
                    })
                else:
                    print("DEBUG: LLM Action Generator - No 'actions' key in response")
                    state["llm_generated_actions"] = []

            except Exception as e:
                print(f"DEBUG: LLM Action Generator - JSON parse error: {str(e)}")
                print(f"DEBUG: LLM Action Generator - Raw response: {response.content}")
                state["llm_generated_actions"] = []

        except Exception as e:
            print(f"DEBUG: LLM Action Generator - Exception: {str(e)}")
            state["error_details"] = f"LLM action generation failed: {str(e)}"
            state["llm_generated_actions"] = []

        return state

    def _find_and_create_submit_action(self, form_html: str) -> Optional[Dict[str, str]]:
        """Find submit button in HTML and create click action for it"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(form_html, 'html.parser')

            # Search for submit buttons in order of priority
            submit_selectors = [
                # Input submit buttons
                soup.find("input", {"type": "submit"}),
                # Button with type submit
                soup.find("button", {"type": "submit"}),
                # Button with submit-related text
                soup.find("button", string=lambda text: text and any(
                    keyword in text.lower() for keyword in ["submit", "continue", "next", "save", "proceed"]
                )),
                # Input button with submit-related value
                soup.find("input", {"type": "button", "value": lambda value: value and any(
                    keyword in value.lower() for keyword in ["submit", "continue", "next", "save", "proceed"]
                )}),
                # Any button with submit-related class
                soup.find(["button", "input"], {"class": lambda classes: classes and any(
                    keyword in " ".join(classes).lower() for keyword in
                    ["submit", "continue", "next", "save", "proceed"]
                )}),
            ]

            # Find the first available submit button
            submit_element = None
            for element in submit_selectors:
                if element:
                    submit_element = element
                    break

            if submit_element:
                # Create selector for the submit button
                tag_name = submit_element.name
                selector_parts = [tag_name]

                # Add ID if available
                if submit_element.get("id"):
                    selector_parts = [f"{tag_name}[id='{submit_element.get('id')}']"]

                # Add type if it's an input
                if tag_name == "input" and submit_element.get("type"):
                    if submit_element.get("id"):
                        selector_parts = [
                            f"input[id='{submit_element.get('id')}'][type='{submit_element.get('type')}']"]
                    else:
                        selector_parts = [f"input[type='{submit_element.get('type')}']"]

                # Add name if available and no ID
                if not submit_element.get("id") and submit_element.get("name"):
                    if tag_name == "input" and submit_element.get("type"):
                        selector_parts = [
                            f"input[type='{submit_element.get('type')}'][name='{submit_element.get('name')}']"]
                    else:
                        selector_parts = [f"{tag_name}[name='{submit_element.get('name')}']"]

                selector = " ".join(selector_parts)

                action = {
                    "selector": selector,
                    "type": "click"
                }

                print(f"DEBUG: Found submit button with selector: {selector}")
                return action

            print("DEBUG: No submit button found in HTML")
            return None

        except Exception as e:
            print(f"DEBUG: Error finding submit button: {e}")
            return None

    def _process_non_form_page(self, state: FormAnalysisState) -> FormAnalysisState:
        """Process non-form pages like task lists, navigation pages"""
        try:
            print("DEBUG: Processing non-form page (task list or navigation page)")

            # Prepare simplified prompt for non-form pages
            prompt = f"""
            # Role 
            You are a web automation expert analyzing HTML pages for actionable elements.

            # HTML Content:
            {state["form_html"]}

            # Instructions:
            Analyze this HTML page and identify clickable elements that represent next steps or actions to take.
            Focus on:
            1. **Task list items with "In progress" status** - these should be clicked
            2. **Links that represent next steps** in a workflow or application process
            3. **Submit buttons or continue buttons**
            4. **Skip items that have "Cannot start yet" or "Completed" status** (unless they need to be revisited)

            # Priority Rules:
            - **HIGHEST PRIORITY**: Links or buttons with "In progress" status
            - **SECOND PRIORITY**: Submit/Continue/Next buttons
            - **THIRD PRIORITY**: Navigation links that advance the process
            - **SKIP**: Disabled buttons, "Cannot start yet" items, or non-actionable elements

            # Response Format (JSON only):
            {{
              "actions": [
                {{
                  "selector": "a[href='/path/to/next/step']",
                  "type": "click"
                }}
              ]
            }}

            **Return ONLY the JSON response, no other text.**
            """

            print(f"DEBUG: Non-form page - Sending prompt to LLM")

            # Use LLM call with workflow_id for logging
            if self._current_workflow_id:
                response = self._invoke_llm([HumanMessage(content=prompt)], self._current_workflow_id)
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])

            print(f"DEBUG: Non-form page - LLM response: {response.content}")

            try:
                # Parse JSON response
                llm_result = robust_json_parse(response.content)

                if "actions" in llm_result and llm_result["actions"]:
                    state["llm_generated_actions"] = llm_result["actions"]
                    print(f"DEBUG: Non-form page - Generated {len(llm_result['actions'])} actions")

                    # Log actions for debugging
                    for i, action in enumerate(llm_result["actions"]):
                        print(
                            f"DEBUG: Non-form action {i + 1}: {action.get('selector', 'unknown')} -> {action.get('type', 'unknown')}")

                    state["messages"].append({
                        "type": "system",
                        "content": f"Generated {len(llm_result['actions'])} actions for non-form page (task list/navigation)"
                    })
                else:
                    print("DEBUG: Non-form page - No actions generated by LLM")
                    state["llm_generated_actions"] = []

            except Exception as e:
                print(f"DEBUG: Non-form page - JSON parse error: {str(e)}")
                print(f"DEBUG: Non-form page - Raw response: {response.content}")
                state["llm_generated_actions"] = []

        except Exception as e:
            print(f"DEBUG: Non-form page processing error: {str(e)}")
            state["error_details"] = f"Non-form page processing failed: {str(e)}"
            state["llm_generated_actions"] = []

        return state

    def _result_saver_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Save results to database"""
        try:
            print("DEBUG: Result Saver - Starting")
            
            # Get step instance
            step = self.step_repo.get_step_by_key(state["workflow_id"], state["step_key"])
            if not step:
                print("DEBUG: Result Saver - Step not found, creating new step")
                step = self.step_repo.create_step(state["workflow_id"], state["step_key"])
            
            # Get existing data to preserve history
            existing_data = step.data or {}
            existing_history = existing_data.get("history", [])
            
            # Create new operation record for history
            new_operation = {
                "processed_at": datetime.utcnow().isoformat(),
                "workflow_id": state["workflow_id"],
                "step_key": state["step_key"],
                "success": not bool(state.get("error_details")),
                "field_count": len(state.get("detected_fields", [])),
                "question_count": len(state.get("field_questions", [])),
                "answer_count": len(state.get("ai_answers", [])),
                "action_count": len(state.get("llm_generated_actions", [])),
                "dummy_data_used_count": len(state.get("dummy_data_usage", []))
            }
            
            # Add error details if present
            if state.get("error_details"):
                new_operation["error_details"] = state["error_details"]
            
            # Append new operation to history (newest at end)
            updated_history = existing_history + [new_operation]
            
            # Prepare the main data structure in the expected format
            save_data = {
                "form_data": state.get("merged_qa_data", []),  # 合并的问答数据
                "actions": state.get("llm_generated_actions", []),  # LLM生成的动作
                "questions": state.get("field_questions", []),  # 原始问题数据
                "dummy_data_usage": state.get("dummy_data_usage", []),  # 虚拟数据使用记录
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "workflow_id": state["workflow_id"],
                    "step_key": state["step_key"],
                    "success": not bool(state.get("error_details")),
                    "field_count": len(state.get("detected_fields", [])),
                    "question_count": len(state.get("field_questions", [])),
                    "answer_count": len(state.get("ai_answers", [])),
                    "action_count": len(state.get("llm_generated_actions", [])),
                    "dummy_data_used_count": len(state.get("dummy_data_usage", []))
                },
                "history": updated_history  # 完整的历史记录
            }
            
            # Add error details to metadata if present
            if state.get("error_details"):
                save_data["metadata"]["error_details"] = state["error_details"]
            
            # Save to database (replace existing data with new structure)
            self.step_repo.update_step_data(step.step_instance_id, save_data)
            
            # Update WorkflowInstance with dummy data usage if any dummy data was used
            dummy_usage = state.get("dummy_data_usage", [])
            if dummy_usage:
                try:
                    # Get workflow instance and update dummy_data_usage field
                    workflow_instance = self.db.query(WorkflowInstance).filter(
                        WorkflowInstance.workflow_instance_id == state["workflow_id"]
                    ).first()
                    
                    if workflow_instance:
                        # Get existing dummy data usage array or initialize empty
                        existing_dummy_usage = workflow_instance.dummy_data_usage or []
                        
                        # Create new records for current processing using the tool function
                        new_records = self._create_workflow_dummy_data(dummy_usage, state["step_key"])
                        
                        # For backward compatibility, also create simple records for all dummy usage
                        simple_records = []
                        for usage in dummy_usage:
                            simple_record = {
                                "processed_at": datetime.utcnow().isoformat(),
                                "step_key": state["step_key"],
                                "question": usage.get("question", ""),
                                "answer": usage.get("answer", ""),
                                "source": usage.get("dummy_data_source", "unknown")
                            }
                            simple_records.append(simple_record)
                        
                        # Append both detailed records and simple records to existing array (incremental update)
                        # This ensures both new format and backward compatibility
                        updated_dummy_usage = existing_dummy_usage + simple_records
                        
                        # Update workflow instance
                        workflow_instance.dummy_data_usage = updated_dummy_usage
                        self.db.commit()
                        
                        print(f"DEBUG: Result Saver - Added {len(simple_records)} dummy data usage records to WorkflowInstance")
                        print(f"DEBUG: Result Saver - {len(new_records)} AI-generated dummy data records created")
                        print(f"DEBUG: Result Saver - Total dummy usage records: {len(updated_dummy_usage)}")
                    else:
                        print(f"DEBUG: Result Saver - Warning: WorkflowInstance {state['workflow_id']} not found for dummy data update")
                        
                except Exception as e:
                    print(f"DEBUG: Result Saver - Error updating WorkflowInstance dummy data usage: {str(e)}")
                    # Don't fail the whole operation if this update fails
            
            # Store saved data in state for reference
            state["saved_step_data"] = save_data
            
            print(f"DEBUG: Result Saver - Saved data for step {state['step_key']} with {len(save_data['form_data'])} form_data items, {len(save_data['actions'])} actions, and {len(updated_history)} history records")
            
            # Mark analysis as complete
            state["analysis_complete"] = True
            
        except Exception as e:
            print(f"DEBUG: Result Saver - Error: {str(e)}")
            state["error_details"] = f"Result saving failed: {str(e)}"
        
        return state

    def _error_handler_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Handle errors"""
        try:
            print(f"DEBUG: Error Handler - Processing error: {state.get('error_details', 'Unknown error')}")
            
            # Get step instance to preserve history
            step = self.step_repo.get_step_by_key(state["workflow_id"], state["step_key"])
            if not step:
                print("DEBUG: Error Handler - Step not found, creating new step")
                step = self.step_repo.create_step(state["workflow_id"], state["step_key"])
            
            # Get existing data to preserve history
            existing_data = step.data or {}
            existing_history = existing_data.get("history", [])
            
            # Create error operation record for history
            error_operation = {
                "processed_at": datetime.utcnow().isoformat(),
                "workflow_id": state["workflow_id"],
                "step_key": state["step_key"],
                "success": False,
                "field_count": len(state.get("detected_fields", [])),
                "question_count": len(state.get("field_questions", [])),
                "answer_count": 0,  # No successful answers in error case
                "action_count": 0,  # No actions in error case
                "error_details": state.get("error_details", "Unknown error")
            }
            
            # Append error operation to history (newest at end)
            updated_history = existing_history + [error_operation]
            
            # Create error response with question structure even on error
            error_questions = []
            
            # Try to extract basic field information even if processing failed
            if state.get("detected_fields"):
                for field in state["detected_fields"]:
                    # Map field type to answer component type
                    answer_type = self._map_field_type_to_answer_type(field.get("type", "text"))
                    
                    # Create answer data with check=0 (no valid answer)
                    answer_data = [{
                        "name": "",
                        "value": "",
                        "check": 0  # No valid answer in error case
                    }]
                    
                    # Apply interrupt logic: since check=0 (no valid answer) and needs_intervention=True,
                    # we should set interrupt=1
                    has_valid_answer = any(item.get("check", 0) == 1 for item in answer_data)
                    needs_intervention = True  # Always true in error case
                    should_interrupt = needs_intervention and not has_valid_answer  # True in error case
                    
                    error_question = {
                        "question": {
                            "data": {
                                "name": field.get("label", field.get("name", "Unknown field"))
                            },
                            "answer": {
                                "type": answer_type,  # Component type from HTML
                                "selector": field.get("selector", ""),
                                "data": answer_data
                            }
                        },
                        "_metadata": {
                            "id": f"error_q_{field.get('name', 'unknown')}_{uuid.uuid4().hex[:8]}",
                            "field_selector": field.get("selector", ""),
                            "field_name": field.get("name", ""),
                            "field_type": field.get("type", "text"),
                            "field_label": field.get("label", ""),
                            "required": field.get("required", False),
                            "options": field.get("options", []),
                            "confidence": 0,
                            "reasoning": f"Error occurred during processing: {state.get('error_details', 'Unknown error')}",
                            "needs_intervention": needs_intervention,
                            "has_valid_answer": has_valid_answer
                        }
                    }
                    
                    # Add interrupt field ONLY if should_interrupt is True (which is always true in error case)
                    if should_interrupt:
                        error_question["question"]["type"] = "interrupt"
                    
                    error_questions.append(error_question)
                    print(f"DEBUG: Error Handler - Created error question for {field.get('name', 'unknown')}: interrupt={should_interrupt}")
            
            # Update state with error questions
            state["merged_qa_data"] = error_questions
            state["llm_generated_actions"] = []
            
            # Save error data in the expected format
            error_save_data = {
                "form_data": error_questions,  # 错误情况下的问答数据
                "actions": [],  # 错误情况下没有动作
                "questions": state.get("field_questions", []),  # 原始问题数据（如果有的话）
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "workflow_id": state["workflow_id"],
                    "step_key": state["step_key"],
                    "success": False,
                    "field_count": len(state.get("detected_fields", [])),
                    "question_count": len(state.get("field_questions", [])),
                    "answer_count": 0,
                    "action_count": 0,
                    "error_details": state.get("error_details", "Unknown error")
                },
                "history": updated_history  # 包含错误记录的历史
            }
            
            # Save error data to database
            self.step_repo.update_step_data(step.step_instance_id, error_save_data)
            
            # Store saved data in state for reference
            state["saved_step_data"] = error_save_data
            
            print(f"DEBUG: Error Handler - Created {len(error_questions)} error questions and saved error data with {len(updated_history)} history records")
            
        except Exception as e:
            print(f"DEBUG: Error Handler - Exception in error handling: {str(e)}")
            # Ensure we have some basic structure even if error handling fails
            state["merged_qa_data"] = []
            state["llm_generated_actions"] = []

        return state

    def _create_workflow_dummy_data(self, dummy_usage: List[Dict[str, Any]], step_key: str) -> List[Dict[str, Any]]:
        """Create formatted dummy data for workflow instance storage
        
        Args:
            dummy_usage: List of dummy data usage records
            step_key: Current step key
            
        Returns:
            List of formatted dummy data records for workflow storage
        """
        workflow_dummy_data = []
        
        for usage in dummy_usage:
            # Only include AI-generated dummy data in workflow storage
            # (provided dummy data from profile_dummy_data is not stored as it's already known)
            if usage.get("dummy_data_source") == "ai_generated":
                workflow_record = {
                    "processed_at": datetime.utcnow().isoformat(),
                    "step_key": step_key,
                    "question": usage.get("question", ""),
                    "answer": usage.get("answer", ""),
                    "field_name": usage.get("field_name", ""),
                    "dummy_data_type": usage.get("dummy_data_type", "unknown"),
                    "confidence": usage.get("confidence", 0),
                    "reasoning": usage.get("reasoning", ""),
                    "source": "ai_generated"
                }
                workflow_dummy_data.append(workflow_record)
        
        return workflow_dummy_data

    async def _invoke_llm_async(self, messages: List, workflow_id: str = None):
        """Async version of LLM invocation (workflow_id kept for logging purposes only)"""
        try:
            # workflow_id is kept for logging but not used in LLM call
            used_workflow_id = workflow_id or self._current_workflow_id
            if used_workflow_id:
                print(f"[workflow_id:{used_workflow_id}] DEBUG: Invoking LLM async")
            # Use async LLM call
            response = await self.llm.ainvoke(messages)
            return response
        except Exception as e:
            print(f"DEBUG: Async LLM invocation error: {str(e)}")
            raise e

    async def _try_answer_with_data_async(self, question: Dict[str, Any], data_source: Dict[str, Any],
                                          source_name: str) -> Dict[str, Any]:
        """Async version of _try_answer_with_data"""
        try:
            # Create prompt for AI (same as sync version)
            prompt = f"""
            # Role 
            You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
            analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
            filling in the form content, and returning it to the front end in a fixed json format.

            # Task
            Based on the user data from {source_name}, determine the appropriate value for this form field:
            
            Field Name: {question['field_name']}
            Field Type: {question['field_type']}
            Field Label: {question['field_label']}
            Field Selector: {question['field_selector']}
            Required: {question['required']}
            Question: {question['question']}
            Available Options: {json.dumps(question.get('options', []), indent=2) if question.get('options') else "No specific options provided"}
            
            ⚠️⚠️⚠️ CRITICAL REMINDER ⚠️⚠️⚠️: If Available Options shows multiple options above, you MUST analyze ALL of them!
            
            # User Data ({source_name}):
            {json.dumps(data_source, indent=2)}
            
            MANDATORY FIRST STEP: COMPREHENSIVE ANALYSIS
            Before attempting to answer, you MUST complete these steps IN ORDER:
            
            STEP 1 - JSON DATA ANALYSIS:
            1. List ALL top-level fields in the JSON above
            2. List ALL nested fields (go deep into objects and arrays)
            3. Identify any fields that could semantically relate to the question
            4. Pay special attention to boolean fields (has*, is*, can*, allow*, enable*)
            5. Look for email-related fields (hasOtherEmail*, additionalEmail*, secondaryEmail*)
            
            STEP 2 - OPTION ANALYSIS (if Available Options are provided):
            1. List ALL available options (both text and value)
            2. For each option, analyze what type of data it expects (boolean, numerical range, text, etc.)
            3. For numerical options (like "3 years or less", "More than 3 years"), identify the threshold values
            4. Determine which option(s) could match the data you found in Step 1
            5. CRITICAL: Your final answer must be the option text or value, NOT the original data value
            
            # Instructions - SEMANTIC UNDERSTANDING AND INTELLIGENT MATCHING
            1. **SEMANTIC UNDERSTANDING**: Understand the MEANING of each data field, not just the field name
            2. **INTELLIGENT MAPPING**: Use AI reasoning to connect data semantics to form questions
            3. **BOOLEAN FIELD INTELLIGENCE**: 
               - For yes/no questions, understand boolean values: true="yes", false="no"
               - Field asking "Do you have X?" + Data "hasX: false" → Answer: "false" or "no" (confidence 85+)
               - Field asking "Are you Y?" + Data "isY: true" → Answer: "true" or "yes" (confidence 85+)
            4. **NUMERICAL COMPARISON AND RANGE MATCHING**:
               - For duration/length questions, compare numerical values intelligently:
               - "5 years" vs "3 years or less" → Does NOT match (5 > 3)
               - "5 years" vs "More than 3 years" → MATCHES (5 > 3) (confidence 90+)
               - "2 years" vs "3 years or less" → MATCHES (2 ≤ 3) (confidence 90+)
               - "2 years" vs "More than 3 years" → Does NOT match (2 ≤ 3)
               - Extract numbers from text and perform logical comparisons
            5. **COMPREHENSIVE OPTION MATCHING**: For radio/checkbox fields with multiple options:
               - Check data value against ALL available options, not just the first one
               - Use logical comparison (numerical, boolean, string matching)
               - Example: Data "5 years" should be checked against both "3 years or less" AND "More than 3 years"
            6. **SEMANTIC MATCHING EXAMPLES** (CRITICAL - STUDY THESE PATTERNS):
               - Question "Do you have another email?" + Data "hasOtherEmailAddresses: false" → Answer: "false" (confidence 90+)
               - Question "Do you have another email address?" + Data "hasOtherEmailAddresses: false" → Answer: "false" (confidence 90+)
               - Question asking about additional/other/secondary email + ANY field containing "hasOther*", "additional*", "secondary*" → Use that boolean value
               - Field about "telephone" + Data "contactInformation.telephoneNumber" → Use the phone number
               - Field about "name" + Data "personalDetails.givenName" → Use the name
               - Question "What is the length of the visa?" with options ["3 years or less", "More than 3 years"] + Data "visaLength: '5 years'" = Answer: "More than 3 years" (confidence 95)
               - Question "What is the length of the visa?" with options ["3 years or less", "More than 3 years"] + Data "visaLength: '2 years'" = Answer: "3 years or less" (confidence 95)
               - Question about duration/period with numerical options + ANY data containing time periods → Compare numerically and match appropriate range
            7. **NESTED DATA SEARCH**: Check nested objects and arrays for relevant data - BE THOROUGH
            8. **CONFIDENCE SCORING - FAVOR SEMANTIC UNDERSTANDING**: 
               - 90-95: Perfect semantic match (hasOtherEmailAddresses:false for "Do you have another email" question, or "5 years" for "More than 3 years")
               - 80-89: Strong semantic match with clear meaning
               - 70-79: Good semantic inference from data structure
               - 50-69: Reasonable inference from context
               - 30-49: Weak match, uncertain
               - 0-29: No suitable semantic match found
            
            ## KEY PRINCIPLE: 
            When options are provided, your answer MUST be one of the option texts/values, NOT the original data value!
            
            ## CRITICAL FINAL STEP - ANSWER VALIDATION:
            Before providing your final JSON response, you MUST:
            1. Re-check that your "answer" field contains EXACTLY one of the available option values or texts
            2. If you found data like "5 years" and determined it matches "More than 3 years", your answer MUST be "More than 3 years" or "moreThanThreeYears"
            3. NEVER put the original data value (like "5 years") in the answer field when options are provided
            4. Double-check your logic: if your reasoning says option X matches, your answer MUST be option X
            
            # Response Format (JSON only):
            {{
                "answer": "value_to_fill_or_empty_string",
                "confidence": 0-100,
                "reasoning": "explanation_of_choice",
                "needs_intervention": true/false,
                "data_source_path": "path.to.data.in.source",
                "field_match_type": "exact|semantic|inferred|none"
            }}
            
            **IMPORTANT**: Return ONLY the JSON response, no other text.
            """

            # Use async LLM call
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Parse response
            try:
                result = robust_json_parse(response.content)

                # Validate required fields
                if not isinstance(result, dict):
                    raise ValueError("Response is not a valid JSON object")

                # Set defaults for missing fields
                result.setdefault("answer", "")
                result.setdefault("confidence", 0)
                result.setdefault("reasoning", "No reasoning provided")
                result.setdefault("needs_intervention", True)
                result.setdefault("data_source_path", "")
                result.setdefault("field_match_type", "none")

                # Add metadata
                result["question_id"] = question["id"]
                result["field_selector"] = question["field_selector"]
                result["field_name"] = question["field_name"]
                result["source_name"] = source_name

                # Enhanced debugging for boolean fields
                field_name = question.get("field_name", "")
                if field_name and any(
                        keyword in field_name.lower() for keyword in ["email", "other", "another", "has"]):
                    print(f"DEBUG: BOOLEAN FIELD PROCESSING - Field: {field_name}")
                    print(f"DEBUG: BOOLEAN FIELD PROCESSING - AI Answer: '{result.get('answer', '')}'")
                    print(f"DEBUG: BOOLEAN FIELD PROCESSING - Confidence: {result.get('confidence', 0)}")
                    print(
                        f"DEBUG: BOOLEAN FIELD PROCESSING - Needs Intervention: {result.get('needs_intervention', True)}")

                return result

            except Exception as parse_error:
                print(f"DEBUG: JSON parsing error in _try_answer_with_data_async: {str(parse_error)}")
                print(f"DEBUG: Raw response: {response.content}")
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "answer": "",
                    "confidence": 0,
                    "reasoning": f"Failed to parse AI response: {str(parse_error)}",
                    "needs_intervention": True,
                    "source_name": source_name
                }

        except Exception as e:
            print(f"DEBUG: Error in _try_answer_with_data_async: {str(e)}")
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error during analysis: {str(e)}",
                "needs_intervention": True,
                "source_name": source_name
            }

    async def _generate_smart_dummy_data_async(self, question: Dict[str, Any], profile: Dict[str, Any],
                                               fallback_result: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of _generate_smart_dummy_data"""
        try:
            # Create intelligent dummy data generation prompt
            prompt = f"""
            # Role
            You are an intelligent form data generator. Generate realistic, contextually appropriate dummy data for form fields.

            # Task
            Generate appropriate dummy data for this form field:
            
            Field Name: {question['field_name']}
            Field Type: {question['field_type']}
            Field Label: {question['field_label']}
            Required: {question['required']}
            Question: {question['question']}
            
            # Context (available user data for context):
            {json.dumps(profile, indent=2, ensure_ascii=False)}
            
            # Previous Analysis Result:
            {json.dumps(fallback_result, indent=2, ensure_ascii=False)}
            
            # Instructions
            1. Generate realistic dummy data that matches the field type and purpose
            2. Use context from available user data when possible (e.g., if user has UK address, generate UK phone number)
            3. Follow common formats and conventions for the field type
            4. Ensure the data is appropriate for the field's purpose
            5. Set confidence based on how well you can generate appropriate data
            
            # Field Type Guidelines:
            - **Phone/Telephone**: Use format appropriate to user's country/region
            - **Email**: Generate realistic email addresses
            - **Names**: Use common names appropriate to user's region
            - **Addresses**: Generate realistic addresses
            - **Dates**: Use reasonable dates for the context
            - **Numbers**: Use appropriate ranges and formats
            - **Text**: Generate contextually appropriate text
            - **Checkboxes/Radio**: Choose most likely option
            - **Select**: Choose most common/appropriate option
            
            # Response Format (JSON only):
            {{
                "answer": "generated_dummy_value",
                "confidence": 40-80,
                "reasoning": "explanation_of_dummy_data_choice",
                "needs_intervention": false,
                "dummy_data_type": "phone|email|name|address|date|number|text|selection|other"
            }}
            
            **IMPORTANT**: Return ONLY the JSON response, no other text.
            """

            # Use async LLM call
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])

            # Parse response
            try:
                result = robust_json_parse(response.content)

                # Validate and set defaults
                if not isinstance(result, dict):
                    raise ValueError("Response is not a valid JSON object")

                result.setdefault("answer", "")
                result.setdefault("confidence", 40)
                result.setdefault("reasoning", "Generated dummy data")
                result.setdefault("needs_intervention", False)
                result.setdefault("dummy_data_type", "other")

                # Add metadata
                result["question_id"] = question["id"]
                result["field_selector"] = question["field_selector"]
                result["field_name"] = question["field_name"]

                # Ensure confidence is reasonable for dummy data (40-80 range)
                if result["confidence"] > 80:
                    result["confidence"] = 80
                elif result["confidence"] < 40:
                    result["confidence"] = 40

                return result

            except Exception as parse_error:
                print(f"DEBUG: JSON parsing error in _generate_smart_dummy_data_async: {str(parse_error)}")
                # Generate basic fallback dummy data if AI parsing fails
                fallback_answer = self._generate_basic_fallback_answer(question)
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "answer": fallback_answer,
                    "confidence": 40,
                    "reasoning": "Generated basic fallback dummy data due to parsing error",
                    "needs_intervention": False,
                    "dummy_data_type": "fallback"
                }

        except Exception as e:
            print(f"DEBUG: Error in _generate_smart_dummy_data_async: {str(e)}")
            # Generate basic fallback dummy data if everything fails
            fallback_answer = self._generate_basic_fallback_answer(question)
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "answer": fallback_answer,
                "confidence": 30,
                "reasoning": f"Generated basic fallback dummy data due to error: {str(e)}",
                "needs_intervention": False,
                "dummy_data_type": "fallback"
            }

    def _generate_basic_fallback_answer(self, question: Dict[str, Any]) -> str:
        """Generate basic fallback dummy data when AI generation fails"""
        field_type = question.get("field_type", "")
        field_name = question.get("field_name", "").lower()

        # Phone/telephone fields
        if "phone" in field_name or "tel" in field_name or field_type == "tel":
            return "07123456789"

        # Email fields  
        elif "email" in field_name or field_type == "email":
            return "user@example.com"

        # Name fields
        elif "name" in field_name and "first" in field_name:
            return "John"
        elif "name" in field_name and "last" in field_name:
            return "Smith"
        elif "name" in field_name:
            return "John Smith"

        # Date fields
        elif field_type in ["date", "datetime-local"]:
            return "1990-01-01"

        # Number fields  
        elif field_type == "number" or "number" in field_name:
            return "123"

        # Address fields
        elif "address" in field_name:
            return "123 Main Street"
        elif "city" in field_name:
            return "London"
        elif "postcode" in field_name or "postal" in field_name:
            return "SW1A 1AA"

        # Default text
        else:
            return "Sample Data"



    async def process_form_async(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any],
                                 profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of process_form using LangGraph workflow"""
        try:
            print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Starting with step_key: {step_key}")
            print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - HTML length: {len(form_html)}")
            print(
                f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Profile data keys: {list(profile_data.keys()) if profile_data else 'None'}")
            print(
                f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Profile dummy data keys: {list(profile_dummy_data.keys()) if profile_dummy_data else 'None'}")

            # Set workflow_id for thread isolation
            self.set_workflow_id(workflow_id)

            # Create initial state
            initial_state = FormAnalysisState(
                workflow_id=workflow_id,
                step_key=step_key,
                form_html=form_html,
                profile_data=profile_data or {},
                profile_dummy_data=profile_dummy_data or {},
                parsed_form=None,
                detected_fields=[],
                field_questions=[],
                ai_answers=[],
                merged_qa_data=[],
                form_actions=[],
                llm_generated_actions=[],
                saved_step_data=None,
                dummy_data_usage=[],
                analysis_complete=False,
                error_details=None,
                messages=[]
            )

            # Process each node asynchronously
            try:
                # HTML Parser node
                initial_state = self._html_parser_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # Field Detector node
                initial_state = self._field_detector_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # Question Generator node
                initial_state = self._question_generator_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # Profile Retriever node
                initial_state = self._profile_retriever_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # AI Answerer node (async)
                initial_state = await self._ai_answerer_node_async(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # QA Merger node
                initial_state = self._qa_merger_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # Action Generator node (traditional, for precise checkbox handling)
                initial_state = self._action_generator_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # LLM Action Generator node (async, for additional actions and submit)
                initial_state = await self._llm_action_generator_node_async(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                # Result Saver node
                initial_state = self._result_saver_node(initial_state)
                if initial_state.get("error_details"):
                    raise Exception(initial_state["error_details"])

                result = initial_state

            except Exception as workflow_error:
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Workflow error: {str(workflow_error)}")
                result = {
                    "error_details": str(workflow_error),
                    "merged_qa_data": [],
                    "llm_generated_actions": [],
                    "messages": []
                }

            print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Workflow completed")
            print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Result keys: {list(result.keys())}")

            # Check for errors
            if result.get("error_details"):
                return {
                    "success": False,
                    "error": result["error_details"],
                    "data": [],
                    "actions": []
                }

            # Return successful result with merged Q&A data
            return {
                "success": True,
                "data": result.get("merged_qa_data", []),  # 返回合并的问答数据
                "actions": result.get("llm_generated_actions", []),  # 返回LLM生成的动作
                "messages": result.get("messages", []),
                "processing_metadata": {
                    "fields_detected": len(result.get("detected_fields", [])),
                    "questions_generated": len(result.get("field_questions", [])),
                    "answers_generated": len(result.get("ai_answers", [])),
                    "actions_generated": len(result.get("llm_generated_actions", [])),
                    "workflow_id": workflow_id,
                    "step_key": step_key
                }
            }

        except Exception as e:
            print(f"DEBUG: process_form_async - Exception: {str(e)}")
            return {
                "success": False,
                "error": f"Form processing failed: {str(e)}",
                "data": [],
                "actions": []
            }

    async def _ai_answerer_node_async(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate AI answers for questions (OPTIMIZED async version with parallel processing)"""
        try:
            print("DEBUG: AI Answerer Async - Starting with parallel processing")

            profile_data = state.get("profile_data", {})
            profile_dummy_data = state.get("profile_dummy_data", {})
            field_questions = state["field_questions"]

            if not field_questions:
                print("DEBUG: AI Answerer Async - No questions to process")
                state["ai_answers"] = []
                state["dummy_data_usage"] = []
                return state

            print(f"DEBUG: AI Answerer Async - Processing {len(field_questions)} questions with advanced optimizations")

            # 🚀 OPTIMIZATION 4: Try early return for exact matches first (fastest)
            early_return_result = self._can_use_early_return(field_questions, profile_data)

            if early_return_result:
                exact_matches = early_return_result["exact_matches"]
                remaining_questions = early_return_result["remaining_questions"]

                print(
                    f"DEBUG: AI Answerer Async - Early return: {len(exact_matches)} exact matches, {len(remaining_questions)} need LLM processing")

                if remaining_questions:
                    # Process remaining questions with batch optimization
                    try:
                        batch_answers = await self._batch_analyze_fields_async(remaining_questions, profile_data,
                                                                               profile_dummy_data)
                        # Combine exact matches with batch results
                        answers = exact_matches + batch_answers
                    except Exception as batch_error:
                        print(f"DEBUG: AI Answerer Async - Batch processing for remaining failed: {str(batch_error)}")
                        # Fallback to individual processing for remaining questions
                        tasks = [self._generate_ai_answer_async(q, profile_data, profile_dummy_data) for q in
                                 remaining_questions]
                        individual_answers = await asyncio.gather(*tasks, return_exceptions=True)
                        valid_individual = [ans for ans in individual_answers if not isinstance(ans, Exception)]
                        answers = exact_matches + valid_individual
                else:
                    # All questions had exact matches
                    answers = exact_matches
            else:
                # 🚀 OPTIMIZATION 2: Try batch processing first (much faster)
                batch_answers = []
                remaining_questions = []

                try:
                    # Attempt batch analysis for all questions
                    batch_answers = await self._batch_analyze_fields_async(field_questions, profile_data,
                                                                           profile_dummy_data)

                    # Check if batch processing was successful
                    if len(batch_answers) == len(field_questions):
                        print(
                            f"DEBUG: AI Answerer Async - Batch processing successful for all {len(batch_answers)} fields")
                        answers = batch_answers
                    else:
                        print(
                            f"DEBUG: AI Answerer Async - Batch processing partial success: {len(batch_answers)}/{len(field_questions)}")
                        # Identify questions that need individual processing
                        processed_field_names = {ans.get("field_name") for ans in batch_answers}
                        remaining_questions = [q for q in field_questions if
                                               q["field_name"] not in processed_field_names]

                        if remaining_questions:
                            print(
                                f"DEBUG: AI Answerer Async - Processing {len(remaining_questions)} remaining questions individually")
                            # 🚀 FALLBACK: Parallel processing for remaining questions
                            tasks = []
                            for question in remaining_questions:
                                task = self._generate_ai_answer_async(question, profile_data, profile_dummy_data)
                                tasks.append(task)

                            individual_answers = await asyncio.gather(*tasks, return_exceptions=True)

                            # Combine batch and individual results
                            all_answers = batch_answers.copy()
                            for answer in individual_answers:
                                if not isinstance(answer, Exception):
                                    all_answers.append(answer)

                            answers = all_answers
                        else:
                            answers = batch_answers

                except Exception as batch_error:
                    print(f"DEBUG: AI Answerer Async - Batch processing failed: {str(batch_error)}")
                    print("DEBUG: AI Answerer Async - Falling back to individual parallel processing")

                    # 🚀 FALLBACK: Original parallel processing approach
                    tasks = []
                    for question in field_questions:
                        task = self._generate_ai_answer_async(question, profile_data, profile_dummy_data)
                        tasks.append(task)

                    # Execute all AI answer generation tasks concurrently
                    start_time = time.time()
                    answers = await asyncio.gather(*tasks, return_exceptions=True)
                    end_time = time.time()

                    print(
                        f"DEBUG: AI Answerer Async - Fallback parallel processing completed in {end_time - start_time:.2f}s")

            # Process results and handle any exceptions
            valid_answers = []
            dummy_usage = []

            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    print(f"DEBUG: AI Answerer Async - Error processing question {i}: {str(answer)}")
                    # Create fallback answer for failed questions
                    question = field_questions[i]
                    fallback_answer = {
                        "question_id": question["id"],
                        "field_selector": question["field_selector"],
                        "field_name": question["field_name"],
                        "answer": "",
                        "confidence": 0,
                        "reasoning": f"Error processing question: {str(answer)}",
                        "needs_intervention": True,
                        "used_dummy_data": False
                    }
                    valid_answers.append(fallback_answer)
                else:
                    valid_answers.append(answer)

                    # Track dummy data usage
                    if answer.get("used_dummy_data", False):
                        dummy_source = answer.get("dummy_data_source", "unknown")
                        dummy_usage.append({
                            "question": field_questions[i].get("question", ""),
                            "field_name": field_questions[i].get("field_name", ""),
                            "answer": answer.get("answer", ""),
                            "dummy_data_source": dummy_source,
                            "dummy_data_type": answer.get("dummy_data_type", "unknown"),
                            "confidence": answer.get("confidence", 0),
                            "reasoning": answer.get("reasoning", "")
                        })

                        if dummy_source == "ai_generated":
                            print(
                                f"DEBUG: AI Answerer Async - Generated smart dummy data for {field_questions[i]['field_name']}: {answer['answer']} (confidence: {answer['confidence']})")
                        else:
                            print(
                                f"DEBUG: AI Answerer Async - Used provided dummy data for {field_questions[i]['field_name']}: {answer['answer']}")
                    else:
                        print(
                            f"DEBUG: AI Answerer Async - Used profile data for {field_questions[i]['field_name']}: confidence={answer['confidence']}")

            state["ai_answers"] = valid_answers
            state["dummy_data_usage"] = dummy_usage

            print(
                f"DEBUG: AI Answerer Async - Generated {len(valid_answers)} answers, {len(dummy_usage)} used dummy data")

        except Exception as e:
            print(f"DEBUG: AI Answerer Async - Error: {str(e)}")
            state["error_details"] = f"AI answer generation failed: {str(e)}"

        return state

    async def _generate_ai_answer_async(self, question: Dict[str, Any], profile: Dict[str, Any],
                                        profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async version of _generate_ai_answer - Modified to only use external dummy data"""
        try:
            # First attempt: try to answer with profile_data (fill_data)
            primary_result = await self._try_answer_with_data_async(question, profile, "profile_data")

            # If we have good confidence and answer from profile data, use it
            if (primary_result["confidence"] >= 20 and  # Lowered threshold to prioritize real data
                    primary_result["answer"] and
                    not primary_result["needs_intervention"]):
                print(
                    f"DEBUG: Using profile_data for field {question['field_name']}: answer='{primary_result['answer']}', confidence={primary_result['confidence']}")
                primary_result["used_dummy_data"] = False
                return primary_result

            # Second attempt: if profile_dummy_data is available and primary answer failed, try external dummy data
            if profile_dummy_data:
                print(
                    f"DEBUG: Trying external dummy data for field {question['field_name']} - primary confidence: {primary_result['confidence']}")
                print(f"DEBUG: Profile dummy data keys: {list(profile_dummy_data.keys())}")
                dummy_result = await self._try_answer_with_data_async(question, profile_dummy_data,
                                                                      "profile_dummy_data")
                print(
                    f"DEBUG: Dummy data result - answer: '{dummy_result['answer']}', confidence: {dummy_result['confidence']}, needs_intervention: {dummy_result['needs_intervention']}")

                # Use dummy data if it's better than primary result OR if primary result has very low confidence
                if (dummy_result["confidence"] > primary_result["confidence"] or
                    (primary_result["confidence"] <= 10 and dummy_result["confidence"] >= 30)) and \
                        dummy_result["answer"] and not dummy_result["needs_intervention"]:
                    dummy_result["used_dummy_data"] = True
                    dummy_result["dummy_data_source"] = "profile_dummy_data"
                    print(
                        f"DEBUG: ✅ Using external dummy data for field {question['field_name']}: answer='{dummy_result['answer']}', confidence={dummy_result['confidence']}")
                    return dummy_result
                else:
                    print(
                        f"DEBUG: ❌ Not using dummy data - dummy confidence: {dummy_result['confidence']}, primary confidence: {primary_result['confidence']}")

            # If primary result has some reasonable confidence, use it even if not perfect
            if primary_result["confidence"] >= 10 and primary_result["answer"]:
                print(
                    f"DEBUG: Using profile_data despite lower confidence for field {question['field_name']}: answer='{primary_result['answer']}', confidence={primary_result['confidence']}")
                primary_result["used_dummy_data"] = False
                return primary_result

            # If all attempts failed, return empty result (no AI generation)
            print(f"DEBUG: No suitable data found for field {question['field_name']}, leaving empty")
            return {
                "question_id": question.get("id", ""),
                "field_selector": question.get("field_selector", ""),
                "field_name": question.get("field_name", ""),
                "answer": "",  # Leave empty if no data available
                "confidence": 0,
                "reasoning": "No suitable data found in profile_data or profile_dummy_data",
                "needs_intervention": True,  # Mark as needing intervention
                "used_dummy_data": False
            }

        except Exception as e:
            # Return empty result on error
            print(f"DEBUG: Exception in _generate_ai_answer_async for {question['field_name']}: {str(e)}")
            return {
                "question_id": question.get("id", ""),
                "field_selector": question.get("field_selector", ""),
                "field_name": question.get("field_name", ""),
                "answer": "",  # Leave empty on error
                "confidence": 0,
                "reasoning": f"Error generating answer: {str(e)}",
                "needs_intervention": True,
                "used_dummy_data": False
            }

    async def _llm_action_generator_node_async(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate actions directly from answer.data for maximum accuracy"""
        try:
            print("DEBUG: LLM Action Generator Async - Starting")

            # Get traditional actions from previous step
            traditional_actions = state.get("form_actions", [])
            merged_qa_data = state.get("merged_qa_data", [])

            print(f"DEBUG: LLM Action Generator Async - Found {len(traditional_actions)} traditional actions")

            if not merged_qa_data:
                print("DEBUG: LLM Action Generator Async - No merged Q&A data available, checking for non-form page")
                # This might be a task list page or other non-form page
                return await self._process_non_form_page_async(state)

            # Generate actions directly from answer.data (most accurate approach)
            final_actions = []

            for item in merged_qa_data:
                question_data = item.get("question", {})
                answer_data = question_data.get("answer", {})
                metadata = item.get("_metadata", {})

                field_type = metadata.get("field_type", "")
                field_name = metadata.get("field_name", "")

                print(f"DEBUG: Processing field '{field_name}' (type: {field_type})")

                # Get data array from answer
                data_array = answer_data.get("data", [])

                for data_item in data_array:
                    if data_item.get("check") == 1:  # Only process checked/selected items
                        selector = data_item.get("selector", "")
                        value = data_item.get("value", "")

                        if not selector:
                            print(f"DEBUG: Skipping item without selector: {data_item}")
                            continue

                        # Determine action type based on field type
                        if field_type in ["radio", "checkbox"]:
                            action_type = "click"
                        elif field_type in ["text", "email", "password", "number", "tel", "url", "date", "time",
                                            "datetime-local", "textarea", "select"]:
                            action_type = "input"
                        else:
                            action_type = "input"  # Default fallback

                        action = {
                            "selector": selector,
                            "type": action_type,
                            "value": value if action_type == "input" else None
                        }

                        final_actions.append(action)
                        print(f"DEBUG: Generated action from data: {action}")

            # Add submit button action
            has_submit = any(
                "submit" in action.get("selector", "").lower() or
                action.get("type") == "submit" or
                ("button" in action.get("selector", "").lower() and "submit" in action.get("selector", "").lower())
                for action in final_actions
            )

            if not has_submit:
                print("DEBUG: LLM Action Generator Async - No submit action found, searching for submit button in HTML")
                submit_action = self._find_and_create_submit_action(state["form_html"])
                if submit_action:
                    final_actions.append(submit_action)
                    print(f"DEBUG: LLM Action Generator Async - Added submit action: {submit_action}")
                else:
                    print("DEBUG: LLM Action Generator Async - No submit button found in HTML")

            # 🚀 OPTIMIZATION: Validate actions before storing
            validated_actions = []
            validation_errors = []
            form_html = state["form_html"]  # Get HTML from state

            for action in final_actions:
                is_valid, error_msg = self._validate_action_selector(action, form_html)
                if is_valid:
                    validated_actions.append(action)
                else:
                    validation_errors.append(f"Action validation failed: {error_msg}")
                    # Try to recover the action
                    recovered_action = self._recover_failed_action(action, form_html)
                    if recovered_action:
                        validated_actions.append(recovered_action)
                        print(f"DEBUG: Recovered action: {recovered_action}")
                    else:
                        print(f"DEBUG: Could not recover action: {action}")

            # Store the validated actions
            state["llm_generated_actions"] = validated_actions
            state["action_validation_errors"] = validation_errors

            print(
                f"DEBUG: LLM Action Generator Async - Validated {len(validated_actions)}/{len(final_actions)} actions")
            if validation_errors:
                print(f"DEBUG: LLM Action Generator Async - {len(validation_errors)} validation errors")

            # Debug: Print all validated actions
            for i, action in enumerate(validated_actions, 1):
                print(
                    f"DEBUG: Final Validated Action {i}: {action.get('selector', 'no selector')} -> {action.get('type', 'no type')} ({action.get('value', 'no value')})")

        except Exception as e:
            print(f"DEBUG: LLM Action Generator Async - Error: {str(e)}")
            state["error_details"] = f"LLM action generation failed: {str(e)}"
            state["llm_generated_actions"] = []

        return state

    def _validate_action_selector(self, action: Dict[str, Any], html_content: str) -> Tuple[bool, str]:
        """🚀 OPTIMIZATION: Validate that action selector exists in HTML"""
        try:
            selector = action.get("selector", "")
            if not selector:
                return False, "Empty selector"

            soup = BeautifulSoup(html_content, 'html.parser')
            elements = soup.select(selector)

            if elements:
                return True, "Valid selector"
            else:
                return False, f"Selector not found: {selector}"

        except Exception as e:
            return False, f"Selector validation error: {str(e)}"

    def _recover_failed_action(self, action: Dict[str, Any], html_content: str) -> Optional[Dict[str, Any]]:
        """🚀 OPTIMIZATION: Try to recover failed action with fallback selectors"""
        try:
            original_selector = action.get("selector", "")
            action_type = action.get("type", "")
            value = action.get("value", "")

            soup = BeautifulSoup(html_content, 'html.parser')

            # Strategy 1: Try common selector variations
            fallback_selectors = []

            if "#value_" in original_selector:
                # For radio/checkbox with value patterns
                if "true" in original_selector:
                    fallback_selectors.extend([
                        "input[value='true']",
                        "input[value='yes']",
                        "input[value='Yes']",
                        "input[type='radio'][value='true']",
                        "input[type='checkbox'][value='true']"
                    ])
                elif "false" in original_selector:
                    fallback_selectors.extend([
                        "input[value='false']",
                        "input[value='no']",
                        "input[value='No']",
                        "input[type='radio'][value='false']",
                        "input[type='checkbox'][value='false']"
                    ])

            # Strategy 2: Try value-based selectors
            if value:
                fallback_selectors.extend([
                    f"input[value='{value}']",
                    f"*[data-value='{value}']",
                    f"input[name*='{value.lower()}']"
                ])

            # Strategy 3: Try type-based selectors
            if action_type == "click":
                fallback_selectors.extend([
                    "input[type='radio']",
                    "input[type='checkbox']",
                    "button[type='submit']"
                ])
            elif action_type == "input":
                fallback_selectors.extend([
                    "input[type='text']",
                    "input[type='email']",
                    "input[type='tel']",
                    "textarea"
                ])

            # Test fallback selectors
            for fallback_selector in fallback_selectors:
                try:
                    elements = soup.select(fallback_selector)
                    if elements:
                        recovered_action = action.copy()
                        recovered_action["selector"] = fallback_selector
                        recovered_action["recovery_method"] = "fallback_selector"
                        recovered_action["original_selector"] = original_selector
                        print(f"DEBUG: Recovered action using fallback selector: {fallback_selector}")
                        return recovered_action
                except:
                    continue

            # Strategy 4: Position-based recovery (last resort)
            if action_type == "click":
                radio_elements = soup.find_all("input", type="radio")
                checkbox_elements = soup.find_all("input", type="checkbox")
                clickable_elements = radio_elements + checkbox_elements

                if clickable_elements:
                    # Use first available clickable element
                    element = clickable_elements[0]
                    if element.get("id"):
                        fallback_selector = f"#{element['id']}"
                    elif element.get("name"):
                        fallback_selector = f"input[name='{element['name']}']"
                    else:
                        fallback_selector = f"input[type='{element.get('type', 'radio')}']"

                    recovered_action = action.copy()
                    recovered_action["selector"] = fallback_selector
                    recovered_action["recovery_method"] = "position_based"
                    recovered_action["original_selector"] = original_selector
                    print(f"DEBUG: Recovered action using position-based selector: {fallback_selector}")
                    return recovered_action

            return None

        except Exception as e:
            print(f"DEBUG: Action recovery error: {str(e)}")
            return None

    async def _process_non_form_page_async(self, state: FormAnalysisState) -> FormAnalysisState:
        """Process non-form pages like task lists, navigation pages (async version)"""
        try:
            print("DEBUG: Processing non-form page (task list or navigation page) - Async")

            # Prepare simplified prompt for non-form pages
            prompt = f"""
            # Role 
            You are a web automation expert analyzing HTML pages for actionable elements.

            # HTML Content:
            {state["form_html"]}

            # Instructions:
            Analyze this HTML page and identify clickable elements that represent next steps or actions to take.
            Focus on:
            1. **Task list items with "In progress" status** - these should be clicked
            2. **Links that represent next steps** in a workflow or application process
            3. **Submit buttons or continue buttons**
            4. **Skip items that have "Cannot start yet" or "Completed" status** (unless they need to be revisited)

            # Priority Rules:
            - **HIGHEST PRIORITY**: Links or buttons with "In progress" status
            - **SECOND PRIORITY**: Submit/Continue/Next buttons
            - **THIRD PRIORITY**: Navigation links that advance the process
            - **SKIP**: Disabled buttons, "Cannot start yet" items, or non-actionable elements

            # Response Format (JSON only):
            {{
              "actions": [
                {{
                  "selector": "a[href='/path/to/next/step']",
                  "type": "click"
                }}
              ]
            }}

            **Return ONLY the JSON response, no other text.**
            """

            print(f"DEBUG: Non-form page async - Sending prompt to LLM")

            # Use async LLM call
            response = await self._invoke_llm_async([HumanMessage(content=prompt)],
                                                    self._current_workflow_id)

            print(f"DEBUG: Non-form page async - LLM response: {response.content}")

            try:
                # Parse JSON response
                llm_result = robust_json_parse(response.content)

                if "actions" in llm_result and llm_result["actions"]:
                    state["llm_generated_actions"] = llm_result["actions"]
                    print(f"DEBUG: Non-form page async - Generated {len(llm_result['actions'])} actions")

                    # Log actions for debugging
                    for i, action in enumerate(llm_result["actions"]):
                        print(
                            f"DEBUG: Non-form async action {i + 1}: {action.get('selector', 'unknown')} -> {action.get('type', 'unknown')}")

                    state["messages"].append({
                        "type": "system",
                        "content": f"Generated {len(llm_result['actions'])} actions for non-form page (task list/navigation) - async"
                    })
                else:
                    print("DEBUG: Non-form page async - No actions generated by LLM")
                    state["llm_generated_actions"] = []

            except Exception as e:
                print(f"DEBUG: Non-form page async - JSON parse error: {str(e)}")
                print(f"DEBUG: Non-form page async - Raw response: {response.content}")
                state["llm_generated_actions"] = []

        except Exception as e:
            print(f"DEBUG: Non-form page async processing error: {str(e)}")
            state["error_details"] = f"Non-form page processing failed: {str(e)}"
            state["llm_generated_actions"] = []

        return state

    async def _batch_analyze_fields_async(self, questions: List[Dict[str, Any]], profile_data: Dict[str, Any],
                                          profile_dummy_data: Dict[str, Any] = None) -> List[
        Dict[str, Any]]:
        """🚀 OPTIMIZATION 2: Batch LLM call with caching - analyze multiple fields in one request"""
        try:
            if not questions:
                return []

            # 🚀 OPTIMIZATION 3: Check cache first
            cache_data = {
                "questions": [{"field_name": q["field_name"], "field_type": q["field_type"], "question": q["question"]}
                              for q in questions],
                "profile_data": profile_data,
                "profile_dummy_data": profile_dummy_data
            }
            cache_key = self._get_cache_key(cache_data)

            # Try to get from cache
            cached_result = self._get_from_cache(cache_key, "batch")
            if cached_result:
                print(f"DEBUG: Batch Analysis - Using cached result for {len(questions)} fields")
                return cached_result

            print(f"DEBUG: Batch Analysis - Processing {len(questions)} fields in single LLM call (cache miss)")

            # Create batch analysis prompt
            fields_data = []
            for i, question in enumerate(questions):
                fields_data.append({
                    "index": i,
                    "field_name": question['field_name'],
                    "field_type": question['field_type'],
                    "field_label": question['field_label'],
                    "question": question['question'],
                    "required": question['required'],
                    "selector": question['field_selector'],
                    "options": question.get('options', [])  # Include options for radio/checkbox fields
                })

            prompt = f"""
            # Role 
            You are a form data analysis expert. Analyze multiple form fields simultaneously and provide answers based on user data.

            # Task
            Analyze ALL the following form fields and provide answers based on the user data:
            
            # Form Fields to Analyze:
            {json.dumps(fields_data, indent=2, ensure_ascii=False)}
            
            ⚠️⚠️⚠️ CRITICAL REMINDER ⚠️⚠️⚠️: For each field above that has "options" array, you MUST check your answer against ALL options in that array!
            
            # User Data (fill_data):
            {json.dumps(profile_data, indent=2, ensure_ascii=False)}
            
            # Profile Dummy Data (backup data):
            {json.dumps(profile_dummy_data, indent=2, ensure_ascii=False) if profile_dummy_data else "None"}
            
            MANDATORY FIRST STEP: COMPREHENSIVE JSON ANALYSIS
            Before attempting to answer any field, you MUST:
            1. List ALL top-level fields in the User Data and Profile Dummy Data above
            2. List ALL nested fields (go deep into objects and arrays)
            3. Identify any fields that could semantically relate to ANY of the form questions
            4. Pay special attention to boolean fields (has*, is*, can*, allow*, enable*)
            5. Look for email-related fields (hasOtherEmail*, additionalEmail*, secondaryEmail*)
            
            # Instructions - SEMANTIC UNDERSTANDING AND INTELLIGENT MATCHING
            1. **SEMANTIC UNDERSTANDING**: Understand the MEANING of each data field, not just the field name
            2. **DATA PRIORITY**: 
               - FIRST: Try to match with User Data (fill_data) using semantic understanding
               - SECOND: If no match in fill_data, try Profile Dummy Data using semantic understanding
               - Use Profile Dummy Data with confidence 80+ if it semantically matches field meaning
            3. **BOOLEAN FIELD INTELLIGENCE**: 
               - For yes/no questions, understand boolean values: true="yes", false="no"
               - Question "Do you have X?" + Data "hasX: false" → Answer: "false" or "no" (confidence 85+)
               - Question "Are you Y?" + Data "isY: true" → Answer: "true" or "yes" (confidence 85+)
            4. **NUMERICAL COMPARISON AND RANGE MATCHING**:
               - For duration/length questions, compare numerical values intelligently:
               - "5 years" vs "3 years or less" → Does NOT match (5 > 3)
               - "5 years" vs "More than 3 years" → MATCHES (5 > 3) (confidence 90+)
               - "2 years" vs "3 years or less" → MATCHES (2 ≤ 3) (confidence 90+)
               - "2 years" vs "More than 3 years" → Does NOT match (2 ≤ 3)
               - Extract numbers from text and perform logical comparisons
            5. **COMPREHENSIVE OPTION MATCHING**: For radio/checkbox fields with multiple options:
               - Check data value against ALL available options, not just the first one
               - Use logical comparison (numerical, boolean, string matching)
               - Example: Data "5 years" should be checked against both "3 years or less" AND "More than 3 years"
            6. **SEMANTIC MATCHING EXAMPLES** (CRITICAL - STUDY THESE PATTERNS):
               
               ## BOOLEAN/YES-NO EXAMPLES:
               - Question "Do you have another email?" + Data "hasOtherEmailAddresses: false" → Answer: "false" (confidence 90+)
               - Question "Do you have another email address?" + Data "hasOtherEmailAddresses: false" → Answer: "false" (confidence 90+)
               - Question asking about additional/other/secondary email + ANY field containing "hasOther*", "additional*", "secondary*" → Use that boolean value
               
               ## DIRECT TEXT EXAMPLES:
               - Field about "telephone" + Data "contactInformation.telephoneNumber" → Use the phone number
               - Field about "name" + Data "personalDetails.givenName" → Use the name
               
               ## NUMERICAL RANGE EXAMPLES (MOST IMPORTANT):
               - Question "What is the length of the visa?" + Data "visaLength: '5 years'" + Options ["3 years or less", "More than 3 years"]:
                 * Step 1: Found data "5 years"
                 * Step 2: Check against "3 years or less" → 5 > 3, so NO MATCH
                 * Step 3: Check against "More than 3 years" → 5 > 3, so MATCH!
                 * Final Answer: "More than 3 years" (confidence 95)
               - Question "What is the length of the visa?" + Data "visaLength: '2 years'" + Options ["3 years or less", "More than 3 years"]:
                 * Step 1: Found data "2 years"  
                 * Step 2: Check against "3 years or less" → 2 ≤ 3, so MATCH!
                 * Final Answer: "3 years or less" (confidence 95)
               
               ## KEY PRINCIPLE: 
               When options are provided, your answer MUST be one of the option texts/values, NOT the original data value!
               
               ## CRITICAL FINAL STEP - ANSWER VALIDATION:
               Before providing your final JSON response, you MUST:
               1. For each field with options, re-check that your "answer" field contains EXACTLY one of the available option values or texts
               2. If you found data like "5 years" and determined it matches "More than 3 years", your answer MUST be "More than 3 years" or "moreThanThreeYears"
               3. NEVER put the original data value (like "5 years") in the answer field when options are provided
               4. Double-check your logic: if your reasoning says option X matches, your answer MUST be option X
            7. **CONFIDENCE SCORING - FAVOR SEMANTIC UNDERSTANDING**: 
               - 90-95: Perfect semantic match in any data source (including numerical range matching)
               - 80-89: Strong semantic match with clear meaning
               - 70-79: Good semantic inference from data structure
               - 50-69: Reasonable inference from context
               - 30-49: Weak match, uncertain
               - 0-29: No good semantic match found
            
            # Response Format (JSON array with one object per field):
            [
                {{
                    "index": 0,
                    "field_name": "field_name",
                    "answer": "value_or_empty_string",
                    "confidence": 0-100,
                    "reasoning": "explanation",
                    "needs_intervention": true/false,
                    "data_source_path": "path.to.data",
                    "field_match_type": "exact|semantic|inferred|none"
                }},
                ...
            ]
            
            **IMPORTANT**: Return ONLY the JSON array, no other text. Process ALL {len(questions)} fields.
            """

            # Single LLM call for all fields
            start_time = time.time()
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            end_time = time.time()

            print(f"DEBUG: Batch Analysis - LLM call completed in {end_time - start_time:.2f}s")

            try:
                results = robust_json_parse(response.content)

                if not isinstance(results, list):
                    raise ValueError("Response is not a list")

                # Validate and format results
                formatted_results = []
                for result in results:
                    if isinstance(result, dict) and "index" in result:
                        index = result["index"]
                        if 0 <= index < len(questions):
                            question = questions[index]
                            # Determine if dummy data was used based on data_source_path or reasoning
                            used_dummy_data = ("contactInformation" in result.get("data_source_path", "") or
                                               "dummy" in result.get("reasoning", "").lower() or
                                               result.get("confidence", 0) >= 70 and result.get("answer", ""))
                            
                            formatted_result = {
                                "question_id": question["id"],
                                "field_selector": question["field_selector"],
                                "field_name": question["field_name"],
                                "answer": result.get("answer", ""),
                                "confidence": result.get("confidence", 0),
                                "reasoning": result.get("reasoning", "Batch analysis result"),
                                "needs_intervention": result.get("needs_intervention", True),
                                "data_source_path": result.get("data_source_path", ""),
                                "field_match_type": result.get("field_match_type", "none"),
                                "used_dummy_data": used_dummy_data,
                                "dummy_data_source": "profile_dummy_data" if used_dummy_data else "",
                                "source_name": "profile_dummy_data" if used_dummy_data else "profile_data"
                            }
                            formatted_results.append(formatted_result)

                # Ensure we have results for all questions
                if len(formatted_results) != len(questions):
                    print(
                        f"DEBUG: Batch Analysis - Warning: Expected {len(questions)} results, got {len(formatted_results)}")
                    # Fill missing results with fallback
                    processed_indices = {r.get("index", -1) for r in results if isinstance(r, dict)}
                    for i, question in enumerate(questions):
                        if i not in processed_indices:
                            fallback_result = {
                                "question_id": question["id"],
                                "field_selector": question["field_selector"],
                                "field_name": question["field_name"],
                                "answer": "",
                                "confidence": 0,
                                "reasoning": "Missing from batch analysis",
                                "needs_intervention": True,
                                "used_dummy_data": False,
                                "source_name": "profile_data"
                            }
                            formatted_results.append(fallback_result)

                print(f"DEBUG: Batch Analysis - Successfully processed {len(formatted_results)} fields")

                # 🚀 OPTIMIZATION 3: Save to cache for future use
                self._save_to_cache(cache_key, formatted_results, "batch")

                return formatted_results

            except Exception as parse_error:
                print(f"DEBUG: Batch Analysis - Parse error: {str(parse_error)}")
                print(f"DEBUG: Raw response: {response.content[:500]}...")
                raise parse_error

        except Exception as e:
            print(f"DEBUG: Batch Analysis - Error: {str(e)}")
            # Fallback to individual processing
            return []

    def _can_use_early_return(self, questions: List[Dict[str, Any]], profile_data: Dict[str, Any]) -> Optional[
        List[Dict[str, Any]]]:
        """🚀 OPTIMIZATION 4: Early return for simple exact matches"""
        try:
            if not questions or not profile_data:
                return None

            print(f"DEBUG: Early Return - Checking {len(questions)} fields for exact matches")

            early_results = []
            exact_matches = 0

            for question in questions:
                field_name = question.get("field_name", "").lower()
                field_type = question.get("field_type", "").lower()

                # Check for exact field name matches in profile data
                exact_match_value = None
                exact_match_path = None

                # Direct field name match
                if field_name in profile_data:
                    exact_match_value = profile_data[field_name]
                    exact_match_path = field_name

                # Check nested structures for common patterns
                if not exact_match_value:
                    for key, value in profile_data.items():
                        if isinstance(value, dict):
                            # Check nested objects
                            if field_name in value:
                                exact_match_value = value[field_name]
                                exact_match_path = f"{key}.{field_name}"
                                break

                            # Check for semantic matches in nested objects
                            if field_name == "telephonenumber" and "telephoneNumber" in value:
                                exact_match_value = value["telephoneNumber"]
                                exact_match_path = f"{key}.telephoneNumber"
                                break
                            elif field_name == "email" and "primaryEmail" in value:
                                exact_match_value = value["primaryEmail"]
                                exact_match_path = f"{key}.primaryEmail"
                                break

                if exact_match_value and str(exact_match_value).strip():
                    # Found exact match
                    result = {
                        "question_id": question["id"],
                        "field_selector": question["field_selector"],
                        "field_name": question["field_name"],
                        "answer": str(exact_match_value),
                        "confidence": 95,  # High confidence for exact matches
                        "reasoning": f"Exact match found at {exact_match_path}",
                        "needs_intervention": False,
                        "data_source_path": exact_match_path,
                        "field_match_type": "exact",
                        "used_dummy_data": False,
                        "source_name": "profile_data"
                    }
                    early_results.append(result)
                    exact_matches += 1
                else:
                    # No exact match found, will need LLM processing
                    early_results.append(None)

            # Only use early return if we have a high percentage of exact matches
            exact_match_ratio = exact_matches / len(questions)
            if exact_match_ratio >= 0.7:  # 70% or more exact matches
                print(
                    f"DEBUG: Early Return - Found {exact_matches}/{len(questions)} exact matches ({exact_match_ratio:.1%})")

                # Fill in the None values with empty results for LLM processing
                final_results = []
                questions_for_llm = []

                for i, result in enumerate(early_results):
                    if result:
                        final_results.append(result)
                    else:
                        questions_for_llm.append(questions[i])

                # Return the exact matches and mark remaining questions for LLM processing
                return {
                    "exact_matches": final_results,
                    "remaining_questions": questions_for_llm
                }

            print(
                f"DEBUG: Early Return - Only {exact_matches}/{len(questions)} exact matches ({exact_match_ratio:.1%}), using full LLM processing")
            return None

        except Exception as e:
            print(f"DEBUG: Early Return - Error: {str(e)}")
            return None
