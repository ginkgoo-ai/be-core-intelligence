import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, TypedDict

from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import SecretStr, BaseModel
from sqlalchemy.orm import Session

from src.database.workflow_repositories import (
    WorkflowInstanceRepository, StepInstanceRepository
)
from src.model.workflow_entities import StepStatus, WorkflowInstance


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
    action_type: str  # input, click, select
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
                print(f"DEBUG: Page belongs to next step {next_step_key}, executing step transition")
                
                # 获取当前步骤实例
                current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
                if current_step:
                    # 1. 完成当前步骤
                    self.step_repo.update_step_status(current_step.step_instance_id, StepStatus.COMPLETED_SUCCESS)
                    current_step.completed_at = datetime.utcnow()
                    print(f"DEBUG: Completed current step {current_step_key}")
                    
                    # 2. 激活下一个步骤
                    next_step = self.step_repo.get_step_by_key(workflow_id, next_step_key)
                    if next_step:
                        self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
                        next_step.started_at = datetime.utcnow()
                        print(f"DEBUG: Activated next step {next_step_key}")
                        
                        # 3. 更新工作流实例的当前步骤
                        from src.database.workflow_repositories import WorkflowInstanceRepository
                        instance_repo = WorkflowInstanceRepository(self.db)
                        instance_repo.update_instance_status(
                            workflow_id, 
                            None,  # 保持当前工作流状态
                            next_step_key  # 更新当前步骤键
                        )
                        print(f"DEBUG: Updated workflow current step to {next_step_key}")
                        
                        # 4. 更新分析结果，指示应该使用下一步骤执行
                        analysis_result.update({
                            "should_use_next_step": True,
                            "next_step_key": next_step_key,
                            "next_step_instance_id": next_step.step_instance_id,
                            "step_transition_completed": True
                        })
                
                # 提交数据库更改
                self.db.commit()
                print(f"DEBUG: Step transition completed and committed")
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

        # Strategy 2: For radio/checkbox fields, look for fieldset legend first
        field_type = element.get("type", "").lower()
        if field_type in ["radio", "checkbox"]:
            # Look for parent fieldset and its legend
            current = element
            for _ in range(5):  # Check up to 5 levels up
                parent = current.find_parent()
                if not parent:
                    break
                if parent.name == "fieldset":
                    legend = parent.find("legend")
                    if legend:
                        legend_text = legend.get_text(strip=True)
                        # Clean up legend text (remove "Required" indicators etc.)
                        import re
                        legend_text = re.sub(r'\(Required\)', '', legend_text, flags=re.IGNORECASE).strip()
                        if legend_text:
                            print(f"DEBUG: _find_field_label - Found fieldset legend for {element.get('name', 'unknown')}: '{legend_text}'")
                            return legend_text
                current = parent

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

        # Strategy 6: Look for nearby text content (like spans, divs with text)
        parent = element.find_parent()
        if parent:
            # Look for text in parent that's not part of other form elements
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
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
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

    def _generate_field_question(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a question for a form field using page context"""
        field_name = field.get("name", "")
        field_label = field.get("label", "")
        field_type = field.get("type", "")
        placeholder = field.get("placeholder", "")
        
        # Generate question text using field-specific information first
        question_text = ""
        
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
        """Generate AI answer for a field question"""
        try:
            # First attempt: try to answer with profile_data only
            primary_result = self._try_answer_with_data(question, profile, "profile_data")
            
            # If primary answer is successful and confident, use it
            if (primary_result["confidence"] >= 50 and 
                primary_result["answer"] and 
                not primary_result["needs_intervention"]):
                return primary_result
            
            # If profile_dummy_data is available and primary answer failed, try dummy data
            if profile_dummy_data:
                print(f"DEBUG: Trying dummy data for field {question['field_name']} - primary confidence: {primary_result['confidence']}")
                dummy_result = self._try_answer_with_data(question, profile_dummy_data, "profile_dummy_data")
                
                # If dummy data provides a better answer, use it and mark as dummy data usage
                if (dummy_result["confidence"] > primary_result["confidence"] and 
                    dummy_result["answer"]):
                    dummy_result["used_dummy_data"] = True
                    dummy_result["dummy_data_source"] = "profile_dummy_data"
                    return dummy_result
            
            # Return the primary result if dummy data didn't help
            primary_result["used_dummy_data"] = False
            return primary_result
                
        except Exception as e:
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error generating answer: {str(e)}",
                "needs_intervention": True,
                "used_dummy_data": False
            }

    def _try_answer_with_data(self, question: Dict[str, Any], data_source: Dict[str, Any], source_name: str) -> Dict[str, Any]:
        """Try to answer a question using a specific data source"""
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
            
            # User Data ({source_name}):
            {json.dumps(data_source, indent=2)}
            
            # Instructions
            1. Find the most appropriate value from the user data for this field
            2. Consider the field type and requirements
            3. If no suitable data is found, set "needs_intervention" to true
            4. Provide confidence level (0-100) based on data quality match
            5. If the data is insufficient or unclear for this specific question, mark it as needing human intervention
            
            # Response Format (JSON only):
            {{
                "answer": "your answer here or empty if no data available",
                "confidence": 85,
                "reasoning": "explanation of why this answer was chosen or why intervention is needed",
                "needs_intervention": false
            }}
            
            # Examples of when needs_intervention should be true:
            - Data doesn't contain relevant information for this field
            - The question requires specific knowledge not in the data
            - The field is required but no matching data exists
            - The question is ambiguous and requires clarification
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                # Try to parse JSON response
                result = json.loads(response.content)
                
                # Determine if intervention is needed based on AI response and confidence
                needs_intervention = result.get("needs_intervention", False)
                confidence = result.get("confidence", 0)
                answer = result.get("answer", "")
                
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
                    "answer": answer,
                    "confidence": confidence,
                    "reasoning": result.get("reasoning", ""),
                    "needs_intervention": needs_intervention
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails - assume intervention needed
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "answer": "",
                    "confidence": 0,
                    "reasoning": "AI response could not be parsed as JSON - intervention needed",
                    "needs_intervention": True
                }
                
        except Exception as e:
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error with {source_name}: {str(e)}",
                "needs_intervention": True
            }

    def _generate_form_action(self, merged_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate form action from merged question-answer data"""
        # Check if we have a valid answer with reasonable confidence
        answer = merged_data.get("answer", "")
        confidence = merged_data.get("confidence", 0)

        # Lower confidence threshold and allow empty answers for some field types
        if not answer and confidence < 10:
            return None
        
        # Get field information
        selector = merged_data["field_selector"]
        field_name = merged_data["field_name"]
        field_type = merged_data.get("field_type", "")

        # Determine action type and value based on field type
        if field_type == "radio":
            # For radio buttons, we need to find the specific radio button to click
            # The selector should point to the specific radio option
            action_type = "click"
            # Find the matching option value to create a more specific selector
            options = merged_data.get("options", [])
            specific_value = None
            for option in options:
                if (str(answer).lower() == str(option.get("value", "")).lower() or
                        str(answer).lower() == str(option.get("text", "")).lower()):
                    specific_value = option.get("value")
                    break

            if specific_value:
                # Create more specific selector for the radio button
                specific_selector = f"input[type='radio'][name='{field_name}'][value='{specific_value}']"
                return {
                    "selector": specific_selector,
                    "action_type": action_type,
                    "value": None,  # For radio/checkbox, clicking is the action
                    "field_name": field_name,
                    "confidence": confidence,
                    "reasoning": merged_data.get("reasoning", ""),
                    "order": hash(field_name) % 1000
                }

        elif field_type == "checkbox":
            # Similar to radio, but for checkboxes
            action_type = "click"
            options = merged_data.get("options", [])
            specific_value = None
            for option in options:
                if (str(answer).lower() == str(option.get("value", "")).lower() or
                        str(answer).lower() == str(option.get("text", "")).lower()):
                    specific_value = option.get("value")
                    break

            if specific_value:
                specific_selector = f"input[type='checkbox'][name='{field_name}'][value='{specific_value}']"
                return {
                    "selector": specific_selector,
                    "action_type": action_type,
                    "value": None,
                    "field_name": field_name,
                    "confidence": confidence,
                    "reasoning": merged_data.get("reasoning", ""),
                    "order": hash(field_name) % 1000
                }

        elif field_type == "select":
            action_type = "select"
            # For select elements, we set the value directly
            return {
                "selector": selector,
                "action_type": action_type,
                "value": answer,
                "field_name": field_name,
                "confidence": confidence,
                "reasoning": merged_data.get("reasoning", ""),
                "order": hash(field_name) % 1000
            }

        else:
            # For text inputs, textarea, etc. - allow even if no answer
            action_type = "input"
        return {
            "selector": selector,
            "action_type": action_type,
                "value": answer,  # Can be empty
            "field_name": field_name,
                "confidence": confidence,
                "reasoning": merged_data.get("reasoning", ""),
                "order": hash(field_name) % 1000
            }

        # If we get here, no action could be generated
        return None

    def _find_answer_for_question(self, question: Dict[str, Any], answers: List[Dict[str, Any]]) -> Optional[
        Dict[str, Any]]:
        """Find the answer for a given question"""
        for answer in answers:
            if answer["field_selector"] == question["field_selector"]:
                return answer
        return None

    def _create_answer_data(self, question: Dict[str, Any], ai_answer: Optional[Dict[str, Any]], needs_intervention: bool) -> List[Dict[str, Any]]:
        """Create answer data array based on field type and AI answer
        
        New logic:
        - If AI answered successfully (not needs_intervention): only store the answer like input format
        - If needs intervention (interrupt): store all options for user selection
        """
        field_type = question["field_type"]
        options = question.get("options", [])
        ai_answer_value = ai_answer.get("answer", "") if ai_answer else ""
        
        if field_type in ["radio", "checkbox", "select"] and options:
            if not needs_intervention and ai_answer_value:
                # AI已回答：只存储答案，类似input格式
                return [{
                    "name": ai_answer_value,
                    "value": ai_answer_value,
                    "check": 1,
                    "selector": question["field_selector"]
                }]
            else:
                # 需要人工干预：存储完整选项列表
                answer_data = []
                
                # Handle multiple answers for checkbox (comma-separated)
                ai_values = []
                if not needs_intervention and ai_answer_value:
                    if field_type == "checkbox" and "," in ai_answer_value:
                        # Multiple selections for checkbox
                        ai_values = [v.strip().lower() for v in ai_answer_value.split(",")]
                    else:
                        # Single selection
                        ai_values = [ai_answer_value.lower()]
                
                for option in options:
                    is_selected = False
                    if ai_values:
                        option_value = str(option.get("value", "")).lower()
                        option_text = str(option.get("text", "")).lower()
                        
                        # Check if any AI value matches this option
                        for ai_val in ai_values:
                            if ai_val == option_value or ai_val == option_text:
                                is_selected = True
                                break
                    
                    # 构建选择器
                    if field_type == "select":
                        selector = question["field_selector"]
                    else:
                        selector = f"input[name='{question['field_name']}'][value='{option.get('value', '')}']"
                    
                    answer_data.append({
                        "name": option.get("text", option.get("value", "")),
                        "value": option.get("value", ""),
                        "check": 1 if is_selected else 0,  # 1 = selected/clicked
                        "selector": selector
                    })
                
                return answer_data
        else:
            # For text inputs, textarea, etc.
            answer_value = ai_answer_value if not needs_intervention else ""
            has_value = bool(answer_value and answer_value.strip())  # Check if there's actually a non-empty value
            
            return [{
                "name": answer_value,
                "value": answer_value,
                "check": 1 if has_value else 0,  # 1 = has input value, 0 = empty
                "selector": question["field_selector"]  # Use the field's selector
            }]


class LangGraphFormProcessor:
    """LangGraph-based form processor for workflow integration"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.instance_repo = WorkflowInstanceRepository(db_session)
        self.step_repo = StepInstanceRepository(db_session)
        self.step_analyzer = StepAnalyzer(db_session)  # 添加 StepAnalyzer
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            base_url=os.getenv("MODEL_BASE_URL"),
            api_key=SecretStr(os.getenv("MODEL_API_KEY")),
            model=os.getenv("MODEL_WORKFLOW_NAME"),
            temperature=0,
        )
        
        # Create workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
    
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
            config = {"configurable": {"thread_id": f"{workflow_id}_{step_key}"}}
            result = self.app.invoke(initial_state, config)
            
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
            print("DEBUG: HTML Parser - Starting")
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
            
            print("DEBUG: HTML Parser - Completed successfully")
            
        except Exception as e:
            print(f"DEBUG: HTML Parser - Error: {str(e)}")
            state["error_details"] = f"HTML parsing failed: {str(e)}"
        
        return state

    def _field_detector_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Detect form fields"""
        try:
            print("DEBUG: Field Detector - Starting")
            
            if not state["parsed_form"]:
                state["error_details"] = "No parsed form available"
                return state
            
            # 修复：从 HTML 字符串重新解析 BeautifulSoup 对象
            html_content = state["parsed_form"]["html_content"]
            form_elements = BeautifulSoup(html_content, 'html.parser')
            
            detected_fields = []
            processed_field_groups = set()  # Track processed radio/checkbox groups
            
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
                    
                    field_info = self.step_analyzer._extract_field_info(element)
                    if field_info:
                        detected_fields.append(field_info)
                        print(f"DEBUG: Field Detector - Found field: {field_info['name']} ({field_info['type']})")
            
            state["detected_fields"] = detected_fields
            print(f"DEBUG: Field Detector - Found {len(detected_fields)} fields (after deduplication)")
            
        except Exception as e:
            print(f"DEBUG: Field Detector - Error: {str(e)}")
            state["error_details"] = f"Field detection failed: {str(e)}"
        
        return state

    def _question_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate questions for form fields"""
        try:
            print("DEBUG: Question Generator - Starting")
            
            # Extract page context for better question generation
            page_context = self.step_analyzer._extract_page_context(state["parsed_form"])
            self.step_analyzer._page_context = page_context
            
            questions = []
            for field in state["detected_fields"]:
                question = self.step_analyzer._generate_field_question(field)
                questions.append(question)
                print(f"DEBUG: Question Generator - Generated question for {field['name']}: {question['question']}")
            
            state["field_questions"] = questions
            print(f"DEBUG: Question Generator - Generated {len(questions)} questions")
            
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
                
                # Track dummy data usage
                if answer.get("used_dummy_data", False):
                    dummy_usage.append({
                        "question": question.get("question", ""),
                        "field_name": question.get("field_name", ""),
                        "answer": answer.get("answer", ""),
                        "dummy_data_source": answer.get("dummy_data_source", "")
                    })
                    print(f"DEBUG: AI Answerer - Used dummy data for {question['field_name']}: {answer['answer']}")
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
            
            # Group questions by question text to merge related fields
            question_groups = {}
            for question in questions:
                question_text = question["question"]
                if question_text not in question_groups:
                    question_groups[question_text] = []
                question_groups[question_text].append(question)
            
            merged_data = []
            
            for question_text, grouped_questions in question_groups.items():
                print(f"DEBUG: Q&A Merger - Processing question group: '{question_text}' with {len(grouped_questions)} fields")
                
                # Determine the primary question (usually the first one or the main input field)
                primary_question = self._find_primary_question(grouped_questions)
                
                # Collect all related fields and their answers
                all_field_data = []
                all_needs_intervention = []
                all_confidences = []
                all_reasonings = []
                
                for question in grouped_questions:
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
                answer_type = self._determine_group_answer_type(grouped_questions)
                
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
                
                # Create merged data structure
                merged_item = {
                    "question": {
                        "data": {
                            "name": question_text
                        },
                        "answer": {
                            "type": answer_type,
                            "selector": primary_question["field_selector"],  # Use primary field's selector
                            "data": all_field_data  # Combined data from all related fields
                        }
                    },
                    # Keep metadata for internal use (using primary question's metadata)
                    "_metadata": {
                        "id": primary_question["id"],
                        "field_selector": primary_question["field_selector"],
                        "field_name": primary_question["field_name"],
                        "field_type": primary_question["field_type"],
                        "field_label": primary_question["field_label"],
                        "required": primary_question["required"],
                        "options": self._combine_all_options(grouped_questions),  # Combined options from all fields
                        "confidence": avg_confidence,
                        "reasoning": combined_reasoning,
                        "needs_intervention": overall_needs_intervention,
                        "has_valid_answer": has_valid_answer,  # Track if answer exists
                        "grouped_fields": [q["field_name"] for q in grouped_questions]  # Track all grouped fields
                    }
                }
                
                # Add interrupt field at question level ONLY if should_interrupt is True
                if should_interrupt:
                    merged_item["question"]["type"] = "interrupt"
                    interrupt_status = "interrupt"
                else:
                    interrupt_status = "normal" if has_valid_answer else "no_answer_but_no_interrupt"
                
                merged_data.append(merged_item)
                print(f"DEBUG: Q&A Merger - Merged question '{question_text}': type={answer_type}, fields={len(grouped_questions)}, status={interrupt_status}")
            
            state["merged_qa_data"] = merged_data
            print(f"DEBUG: Q&A Merger - Created {len(merged_data)} merged question groups from {len(questions)} original questions")
            
        except Exception as e:
            print(f"DEBUG: Q&A Merger - Error: {str(e)}")
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
                
                # Skip if question needs intervention (interrupt)
                if question_data.get("type") == "interrupt":
                    print(f"DEBUG: Action Generator - Skipping {metadata.get('field_name', 'unknown')} due to interrupt")
                    continue
                
                answer_data = question_data.get("answer", {})
                
                # Create a compatible data structure for the existing action generator
                compatible_item = {
                    "field_selector": metadata.get("field_selector", ""),
                    "field_name": metadata.get("field_name", ""),
                    "field_type": metadata.get("field_type", ""),
                    "options": metadata.get("options", []),
                    "confidence": metadata.get("confidence", 0),
                    "reasoning": metadata.get("reasoning", ""),
                    "answer": self._extract_answer_from_data(answer_data)
                }
                
                action = self.step_analyzer._generate_form_action(compatible_item)
                if action:
                    actions.append(action)
                    print(f"DEBUG: Action Generator - Generated action for {metadata.get('field_name', 'unknown')}: {action['action_type']}")
            
            # Sort actions by order
            actions.sort(key=lambda x: x.get("order", 0))
            
            state["form_actions"] = actions
            print(f"DEBUG: Action Generator - Generated {len(actions)} actions")
            
        except Exception as e:
            print(f"DEBUG: Action Generator - Error: {str(e)}")
            state["error_details"] = f"Action generation failed: {str(e)}"
        
        return state

    def _extract_answer_from_data(self, answer_data: Dict[str, Any]) -> str:
        """Extract answer value from answer data structure"""
        data_array = answer_data.get("data", [])
        
        # Find the checked item
        for item in data_array:
            if item.get("check") == 1:
                return item.get("value", item.get("name", ""))
        
        # If no checked item, return empty string
        return ""

    def _llm_action_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Use LLM to generate actions in the format required by graph.py"""
        try:
            print("DEBUG: LLM Action Generator - Starting")
            
            # Prepare enhanced prompt with HTML and profile data
            prompt = f"""
            # Role 
            You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
            analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
            filling in the form content, and returning it to the front end in a fixed json format. only json information is needed

            # HTML Content:
            {state["form_html"]}

            # User Profile Data:
            {json.dumps(state["profile_data"], indent=2, ensure_ascii=False)}

            # Response json format (EXACTLY like this):
            {{
              "actions": [
                {{
                  "selector": "input[type='text'][name='username']",
                  "type": "input",
                  "value": "张三"
                }},
                {{
                  "selector": "input[type='radio'][name='gender'][value='male']",
                  "type": "click"
                }},
                {{
                  "selector": "select[name='country']",
                  "type": "input",
                  "value": "china"
                }},
                {{
                  "selector": "a[href='/next-page']",
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
            1. Analyze the HTML form elements
            2. Match form fields with appropriate data from the user profile
            3. Generate CSS selectors for each form element
            4. For input fields (text, email, password, etc.): use "type": "input" and provide "value"
            5. For radio buttons and checkboxes: use "type": "click" (no value needed)
            6. For select dropdowns: use "type": "input" and provide "value"
            7. For submit buttons: use "type": "click" (no value needed)

            ## For Task List Pages (Government/Application Pages):
            8. Look for task lists with status indicators (like "In progress", "Cannot start yet", "Completed")
            9. For tasks with "In progress" status that have clickable links: generate "type": "click" actions
            10. For tasks with "Cannot start yet" status: do NOT generate actions (skip them)
            11. For completed tasks: generally skip unless specifically needed
            12. Use the exact href attribute value in the selector: a[href="exact-path"]

            ## Priority Rules:
            - If the page contains a task list with "In progress" items, prioritize clicking those links
            - If the page is a traditional form, prioritize filling form fields
            - If both exist, handle both types of actions
            - Always use precise CSS selectors that will uniquely identify the element

            ## CSS Selector Examples:
            - For links: a[href="/STANDALONE_ACCOUNT_REGISTRATION/3434-0940-8939-9272/identity"]
            - For form inputs: input[type="text"][name="firstName"]
            - For radio buttons: input[type="radio"][name="gender"][value="male"]
            - For checkboxes: input[type="checkbox"][name="agree"][value="yes"]
            - For select dropdowns: select[name="country"]
            - For buttons: button[type="submit"] or input[type="submit"]

            # Analysis Steps:
            1. First, determine if this is a task list page or a traditional form
            2. If task list: identify tasks with "In progress" status and clickable links
            3. If traditional form: identify form fields that can be filled with profile data
            4. Generate appropriate actions based on the page type
            5. Return ONLY the JSON response, no other text

            Please provide the complete JSON response for this page:
            """

            print(f"DEBUG: LLM Action Generator - Sending enhanced prompt to LLM")
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            print(f"DEBUG: LLM Action Generator - Received response: {response.content}")

            try:
                # Try to parse JSON response
                llm_result = json.loads(response.content)
                
                if "actions" in llm_result:
                    # Store the LLM-generated actions in the dedicated field
                    state["llm_generated_actions"] = llm_result["actions"]
                    print(f"DEBUG: LLM Action Generator - Successfully parsed {len(llm_result['actions'])} actions")
                    
                    # Log the types of actions generated for debugging
                    action_types = {}
                    for action in llm_result["actions"]:
                        action_type = action.get("type", "unknown")
                        action_types[action_type] = action_types.get(action_type, 0) + 1
                    print(f"DEBUG: LLM Action Generator - Action types: {action_types}")
                    
                    state["messages"].append({
                        "type": "system",
                        "content": f"LLM generated {len(llm_result['actions'])} actions successfully. Types: {action_types}"
                    })
                else:
                    print("DEBUG: LLM Action Generator - No 'actions' key in response")
                    state["llm_generated_actions"] = []
                    
            except json.JSONDecodeError as e:
                print(f"DEBUG: LLM Action Generator - JSON parse error: {str(e)}")
                print(f"DEBUG: LLM Action Generator - Raw response: {response.content}")
                
                # Fallback: try to extract JSON from response
                content = response.content.strip()
                if content.startswith('```json'):
                    content = content[7:]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()
                
                try:
                    llm_result = json.loads(content)
                    if "actions" in llm_result:
                        state["llm_generated_actions"] = llm_result["actions"]
                        print(f"DEBUG: LLM Action Generator - Successfully parsed actions after cleanup")
                    else:
                        state["llm_generated_actions"] = []
                except:
                    state["llm_generated_actions"] = []
                    print("DEBUG: LLM Action Generator - Failed to parse JSON even after cleanup")

        except Exception as e:
            print(f"DEBUG: LLM Action Generator - Exception: {str(e)}")
            state["error_details"] = f"LLM action generation failed: {str(e)}"
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
                        
                        # Create new records for current processing
                        new_records = []
                        for usage in dummy_usage:
                            new_record = {
                                "processed_at": datetime.utcnow().isoformat(),
                                "step_key": state["step_key"],
                                "question": usage.get("question", ""),
                                "answer": usage.get("answer", "")
                            }
                            new_records.append(new_record)
                        
                        # Append new records to existing array (incremental update)
                        updated_dummy_usage = existing_dummy_usage + new_records
                        
                        # Update workflow instance
                        workflow_instance.dummy_data_usage = updated_dummy_usage
                        self.db.commit()
                        
                        print(f"DEBUG: Result Saver - Added {len(new_records)} dummy data usage records to WorkflowInstance (total: {len(updated_dummy_usage)})")
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