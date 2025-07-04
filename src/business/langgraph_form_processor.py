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

from src.business.cross_page_data_cache import CrossPageDataCache
from src.database.workflow_repositories import (
    StepInstanceRepository
)
from src.model.workflow_entities import StepStatus, WorkflowInstance


# Enhanced context analyzer removed - using built-in semantic analysis instead
def clean_llm_response_array(response_content: str) -> str:
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
    return content

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
    consistency_issues: Optional[List[Dict[str, str]]]  # 🚀 NEW: Consistency issues
    conditional_context: Optional[Dict[str, Any]]  # 🚀 NEW: Universal conditional context
    # 🚀 NEW: Conditional field analysis state variables
    field_selections: Optional[Dict[str, str]]  # Map of field_name to selected_value
    active_field_groups: Optional[List[Dict[str, Any]]]  # List of activated field groups
    conditionally_filtered_questions: Optional[List[Dict[str, Any]]]  # Questions after conditional filtering
    # 🚀 NEW: Semantic question analysis state variables
    semantic_question_groups: Optional[List[Dict[str, Any]]]  # AI-analyzed question groups
    question_semantic_analysis: Optional[Dict[str, Any]]  # AI semantic analysis results
    semantically_filtered_questions: Optional[List[Dict[str, Any]]]  # Questions after semantic filtering
    # 🚀 NEW: Cross-page analysis state variables
    cross_page_analysis: Optional[Dict[str, Any]]  # Cross-page data analysis results
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
    """Step analyzer for analyzing which step the current page belongs to"""

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
        Analyze whether the current page belongs to the current step or the next step

        Improved logic:
        - Get context information for both current step and next step
        - Use LLM to compare which step the page content better matches
        - If it belongs to the next step, complete current step and activate next step
        """
        try:
            # Extract page questions
            page_analysis = self._extract_page_questions(html_content)

            # Get current step context
            current_step_context = self._get_step_context(workflow_id, current_step_key)

            # Get next step context
            next_step_key = self._find_next_step(workflow_id, current_step_key)
            next_step_context = None
            if next_step_key:
                next_step_context = self._get_step_context(workflow_id, next_step_key)

            # Use LLM for comparative analysis
            analysis_result = self._analyze_with_llm(
                page_analysis=page_analysis,
                current_step_context=current_step_context,
                next_step_context=next_step_context,
                current_step_key=current_step_key,
                next_step_key=next_step_key
            )

            # If page belongs to next step, execute step transition
            if analysis_result.get("belongs_to_next_step", False) and next_step_key:
                print(
                    f"[workflow_id:{workflow_id}] DEBUG: Page belongs to next step {next_step_key}, executing step transition")

                # Get current step instance
                current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
                if current_step:
                    # 1. Complete current step
                    self.step_repo.update_step_status(current_step.step_instance_id, StepStatus.COMPLETED_SUCCESS)
                    current_step.completed_at = datetime.now(datetime.UTC)
                    print(f"[workflow_id:{workflow_id}] DEBUG: Completed current step {current_step_key}")

                    # 2. Activate next step
                    next_step = self.step_repo.get_step_by_key(workflow_id, next_step_key)
                    if next_step:
                        self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
                        next_step.started_at = datetime.now(datetime.UTC)
                        print(f"[workflow_id:{workflow_id}] DEBUG: Activated next step {next_step_key}")

                        # 3. Update workflow instance's current step
                        from src.database.workflow_repositories import WorkflowInstanceRepository
                        instance_repo = WorkflowInstanceRepository(self.db)
                        instance_repo.update_instance_status(
                            workflow_id,
                            None,  # Keep current workflow status
                            next_step_key  # Update current step key
                        )
                        print(f"[workflow_id:{workflow_id}] DEBUG: Updated workflow current step to {next_step_key}")

                        # 4. Update analysis result to indicate next step should be used
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
        """Extract page questions and context information"""
        soup = BeautifulSoup(html_content, 'html.parser')

        analysis = {
            "page_title": "",
            "form_title": "",
            "main_heading": "",
            "questions": [],  # All questions on the page
            "form_elements": []
        }

        # Extract page title
        title_tag = soup.find("title")
        if title_tag:
            analysis["page_title"] = title_tag.get_text(strip=True)

        # Extract main heading
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                analysis["main_heading"] = heading.get_text(strip=True)
                break

        # Extract form title
        form = soup.find("form")
        if form:
            legend = form.find("legend")
            if legend:
                analysis["form_title"] = legend.get_text(strip=True)

            # Extract form elements and corresponding questions
            for element in form.find_all(["input", "select", "textarea"]):
                # Get field label
                label = self._find_field_label(element)

                # Build question
                question = {
                    "field_name": element.get("name", ""),
                    "field_type": element.get("type", "text"),
                    "question_text": label or element.get("placeholder", ""),
                    "required": element.get("required") is not None,
                    "options": self._extract_field_options(element)
                }

                analysis["questions"].append(question)

                # Save form element information
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
        """提取字段的选项 - 保持原始的value和text结构"""
        options = []

        if element.name == "select":
            for option in element.find_all("option"):
                option_value = option.get("value", "")
                option_text = option.get_text(strip=True)
                if option_value or option_text:
                    # 🚀 FIXED: For regular select fields, keep original value and text structure
                    # This allows different handling for regular select vs autocomplete fields
                    options.append({
                        "value": option_value or option_text,  # Keep original value
                        "text": option_text or option_value,  # Keep original text
                        "original_value": option_value  # Keep original value for reference
                    })
                    print(
                        f"DEBUG: _extract_field_options - SELECT option: text='{option_text}', value='{option_value}'")
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
            print(
                f"DEBUG: _extract_field_info - Element attributes: name={element.get('name', '')}, id={element.get('id', '')}, class={element.get('class', '')}")

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

            # Generate unique ID for field tracking
            field_id = element.get("id", "") or element.get("name", "") or self._generate_selector(element)

            # Get the label first so we can use it for required detection
            field_label = self._find_field_label(element)
            
            # Enhanced required detection: check HTML attribute OR label text OR aria-required OR fieldset legend
            html_required = element.get("required") is not None
            aria_required = element.get("aria-required") == "true"
            label_required = (field_label and "(Required)" in field_label) or \
                           (field_label and "Required" in field_label and 
                            ("*" in field_label or field_label.strip().endswith("Required")))
            
            # Check fieldset legend for Required indication
            fieldset_required = False
            fieldset_parent = element.find_parent("fieldset")
            if fieldset_parent:
                legend = fieldset_parent.find("legend")
                if legend:
                    legend_text = legend.get_text(strip=True)
                    if "(Required)" in legend_text or "Required" in legend_text:
                        fieldset_required = True
            
            is_required = html_required or aria_required or label_required or fieldset_required
            
            if is_required:
                print(f"DEBUG: _extract_field_info - Field '{element.get('name', '')}' marked as REQUIRED: html_required={html_required}, aria_required={aria_required}, label_required={label_required}, fieldset_required={fieldset_required}, label='{field_label}'")
            
            field_info = {
                "id": field_id,  # Add unique ID for tracking
                "name": element.get("name", ""),
                "type": field_type,
                "selector": self._generate_selector(element),
                "required": is_required,  # Enhanced required detection
                "label": field_label,
                "placeholder": element.get("placeholder", ""),
                "value": element.get("value", ""),
                "options": []
            }

            print(f"DEBUG: _extract_field_info - Basic field info: {field_info}")

            # Handle different field types and their options
            if field_type == "select":
                # 🚀 Use the improved _extract_field_options method that prioritizes text over value
                options = self._extract_field_options(element)
                field_info["options"] = options
                print(f"DEBUG: _extract_field_info - Select options (using text priority): {options}")

            elif field_type == "radio":
                # For radio buttons, find all related radio buttons with same name
                radio_name = element.get("name", "")
                if radio_name:
                    # Find the container and look for related radio buttons
                    # Search in a wider scope to find all radio buttons with same name
                    container = element.find_parent()
                    search_scope = container

                    # If container is too small, search in document root
                    while search_scope and len(
                            search_scope.find_all('input', {'type': 'radio', 'name': radio_name})) < 2:
                        search_scope = search_scope.find_parent()
                        if not search_scope:
                            # Search in the entire document
                            search_scope = element
                            while search_scope.parent:
                                search_scope = search_scope.parent
                            break

                    related_radios = search_scope.find_all('input', {'type': 'radio',
                                                                     'name': radio_name}) if search_scope else []
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

            print(
                f"DEBUG: _extract_field_info - Validation: has_name={has_name}, has_label={has_label}, has_selector={has_selector}")

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

                    # Strategy 1: Look for details/summary structure that contains this field
                    details_elements = soup.find_all('details')
                    for details in details_elements:
                        # Check if this details section contains our field
                        field_inputs = details.find_all('input', {'name': field_name})
                        if field_inputs:
                            summary = details.find('summary')
                            if summary:
                                # Extract text from summary, removing aria-controls and other attributes
                                summary_text = summary.get_text(strip=True)
                                if summary_text and len(summary_text) > 3:
                                    print(f"DEBUG: Found details/summary for {field_name}: '{summary_text}'")
                                    return summary_text

                    # Strategy 2: Look for fieldset legend that contains this field
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

                    # Strategy 3: Look for heading or label before the field group
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
                                        # Check if this heading is related to the field context
                                        if (heading_text and len(heading_text) > 5 and
                                                any(keyword in heading_text.lower() for keyword in
                                                    ['email', 'contact', 'address', 'phone', 'telephone', 'another',
                                                     'parent', 'family', 'details'])):
                                            print(f"DEBUG: Found related heading for {field_name}: '{heading_text}'")
                                            return heading_text

                                current = parent

                    # Strategy 3b: Simple form-external question search
                    if field_name:
                        first_field = soup.find('input', {'name': field_name})
                        if first_field:
                            # Find the form containing this field
                            form_element = first_field.find_parent('form')
                            if form_element:
                                # Look for all span/div/p elements that come before the form
                                all_elements = soup.find_all(['span', 'p', 'div', 'h1', 'h2', 'h3'])
                                form_position = list(soup.descendants).index(form_element) if form_element in soup.descendants else -1
                                
                                for elem in all_elements:
                                    try:
                                        elem_position = list(soup.descendants).index(elem) if elem in soup.descendants else -1
                                        # If element comes before form in document order
                                        if elem_position < form_position and elem_position != -1:
                                            text = elem.get_text(strip=True)
                                            # Look for question patterns, but exclude text that contains option words
                                            if (text and 5 < len(text) < 80 and
                                                (text.endswith('?') or
                                                 any(word in text.lower() for word in ['are you', 'do you', 'have you'])) and
                                                # Exclude text that contains common option words (indicates mixed content)
                                                not any(option_word in text.lower() for option_word in ['yes', 'no', 'true', 'false', 'submit', 'next', 'continue'])):
                                                print(f"DEBUG: Found form-external question: '{text}'")
                                                return text
                                            
                                            # If text contains option words, try to extract just the question part
                                            elif (text and 5 < len(text) < 150 and
                                                  (text.endswith('?') or any(word in text.lower() for word in ['are you', 'do you', 'have you'])) and
                                                  any(option_word in text.lower() for option_word in ['yes', 'no', 'true', 'false'])):
                                                # Try to split and get the question part
                                                sentences = text.split('?')
                                                if len(sentences) >= 2:
                                                    question_part = sentences[0] + '?'
                                                    if (question_part and 5 < len(question_part.strip()) < 80 and
                                                        any(word in question_part.lower() for word in ['are you', 'do you', 'have you'])):
                                                        print(f"DEBUG: Found extracted question part: '{question_part.strip()}'")
                                                        return question_part.strip()
                                    except (ValueError, AttributeError):
                                        continue

                    # Strategy 4: Look for descriptive text near the field (Enhanced)
                    if field_name:
                        first_field = soup.find('input', {'name': field_name})
                        if first_field:
                            # Strategy 4a: Look for preceding span/p/div with question patterns
                            for sibling in first_field.find_all_previous(['span', 'p', 'div', 'h1', 'h2', 'h3']):
                                text = sibling.get_text(strip=True)
                                # Enhanced pattern matching for question-like text
                                if (text and 5 < len(text) < 100 and  # Question length range
                                    (text.endswith('?') or  # Direct question
                                     any(pattern in text.lower() for pattern in
                                         ['are you', 'do you', 'have you', 'will you', 'can you', 'where are',
                                          'what is', 'which', 'how', 'when', 'where', 'who', 'why']))):
                                    print(f"DEBUG: Found question text for {field_name}: '{text}'")
                                    return text
                            
                            # Strategy 4b: Look for text within reasonable proximity (broader search)
                            form_parent = first_field.find_parent('form')
                            if form_parent and form_parent.find_parent():
                                # Search in form's parent container for preceding question text
                                form_container = form_parent.find_parent()
                                for elem in form_container.find_all(['span', 'p', 'div', 'h1', 'h2', 'h3']):
                                    # Check if this element comes before the form
                                    try:
                                        if elem.sourceline and form_parent.sourceline and elem.sourceline < form_parent.sourceline:
                                            text = elem.get_text(strip=True)
                                            if (text and 5 < len(text) < 100 and
                                                (text.endswith('?') or
                                                 any(pattern in text.lower() for pattern in
                                                     ['are you', 'do you', 'have you', 'will you', 'can you', 'where are',
                                                      'what is', 'which', 'how', 'when', 'where', 'who', 'why']))):
                                                print(f"DEBUG: Found form-external question for {field_name}: '{text}'")
                                                return text
                                    except (AttributeError, TypeError):
                                        # Fallback: check by position in document
                                        pass
                            
                            # Strategy 4c: Original descriptive text search (fallback)
                            for sibling in first_field.find_all_previous(['p', 'div', 'span', 'label']):
                                text = sibling.get_text(strip=True)
                                if (text and 20 < len(text) < 150 and  # Reasonable question length
                                        any(keyword in text.lower() for keyword in
                                            ['email', 'contact', 'address', 'phone', 'telephone', 'another', 'do you',
                                             'have you', 'parent', 'family', 'details'])):
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
                elif any(keyword in question.lower() for keyword in ['parent', 'family']):
                    if 'unknown' in question.lower():
                        return f"What if I do not have my parents' details?"
                    else:
                        return f"Please provide your {question.lower()}"
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
            if any(word in question_text.lower() for word in
                   ['what', 'which', 'how', 'when', 'where', 'who', 'choose', 'select', 'enter']):
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

    def _generate_enhanced_selector(self, question: Dict[str, Any], option_value: str = None) -> str:
        """Generate enhanced CSS selector with element type and attributes
        
        Examples:
        - input[id="mandatoryDocuments_travelDocRef-TUR-Elif Kaya_0"]
        - input[type="radio"][name="warCrimesInvolvement"][value="false"]
        """
        try:
            field_type = question.get("field_type", "")
            field_name = question.get("field_name", "")
            original_selector = question.get("field_selector", "")
            
            # Extract element type from field type
            if field_type in ["radio", "checkbox"]:
                element_type = "input"
            elif field_type == "textarea":
                element_type = "textarea"
            elif field_type in ["select", "select-one", "select-multiple"]:
                element_type = "select"
            elif field_type in ["text", "email", "password", "number", "tel", "url", "date", "time", "datetime-local"]:
                element_type = "input"
            else:
                element_type = "input"  # Default fallback
            
            # Try to extract ID pattern from original selector
            base_element_id = ""
            if original_selector.startswith("#"):
                base_element_id = original_selector[1:]  # Remove #
            elif "id=" in original_selector:
                # Extract from attribute selector: input[id="someId"]
                import re
                id_match = re.search(r'id=["\']([^"\']*)["\']', original_selector)
                if id_match:
                    base_element_id = id_match.group(1)
            
            # 🚀 CRITICAL FIX: For radio/checkbox, use the original base ID directly
            # Each radio option has its own unique ID in the HTML, don't generate new ones
            element_id = ""
            if field_type in ["radio", "checkbox"] and option_value:
                if base_element_id:
                    # For radio/checkbox fields, the base_element_id IS the correct ID for this specific option
                    # Don't modify it by adding option_value - each option has its own unique ID
                    element_id = base_element_id
                else:
                    # Only generate ID if no base ID exists
                    if option_value in ["true", "false"]:
                        element_id = f"{field_name}_{option_value}"
                    elif field_name.endswith("[0]"):
                        # For array fields like mandatoryDocuments[0], use base name + value + index
                        base_name = field_name.replace("[0]", "")
                        element_id = f"{base_name}_{option_value}_0"
                    else:
                        # For other cases, use field_name + option_value pattern
                        element_id = f"{field_name}_{option_value}".replace("[", "_").replace("]", "")
            elif base_element_id:
                # For non-radio/checkbox fields, use the original ID as-is
                element_id = base_element_id
            
            # Generate enhanced selector based on available information
            if element_id:
                # Use ID-based selector with element type
                enhanced_selector = f'{element_type}[id="{element_id}"]'
            elif field_name and option_value:
                # Generate attribute-based selector for radio/checkbox with specific value
                enhanced_selector = f'{element_type}[type="{field_type}"][name="{field_name}"][value="{option_value}"]'
            elif field_name:
                # Generate attribute-based selector for the field
                if field_type in ["radio", "checkbox"]:
                    enhanced_selector = f'{element_type}[type="{field_type}"][name="{field_name}"]'
                else:
                    enhanced_selector = f'{element_type}[name="{field_name}"]'
            else:
                # Fallback to original selector with element type
                if original_selector.startswith("#") or original_selector.startswith("."):
                    # Keep CSS shorthand but add element type
                    enhanced_selector = f'{element_type}{original_selector}'
                else:
                    enhanced_selector = original_selector
                
            print(f"DEBUG: Enhanced selector - Field: '{field_name}', Type: '{field_type}', Option: '{option_value}' -> '{enhanced_selector}'")
            return enhanced_selector
            
        except Exception as e:
            print(f"DEBUG: Error generating enhanced selector: {str(e)}")
            return question.get("field_selector", "")

    def _create_answer_data(self, question: Dict[str, Any], ai_answer: Optional[Dict[str, Any]],
                            needs_intervention: bool) -> List[Dict[str, Any]]:
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

        if field_type in ["radio", "checkbox", "select", "autocomplete"] and options:
            # 🚀 CRITICAL FIX: Consider needs_intervention parameter for option-based fields
            if has_ai_answer and not needs_intervention:  # AI provided an answer and no intervention needed
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
                        original_selector = question.get("field_selector", "")

                        # 🚀 ENHANCED: Use enhanced selector generation
                        if field_type == "checkbox" and len(options) == 1:
                            # Single checkbox uses original selector enhanced
                            correct_selector = self._generate_enhanced_selector(question)
                        else:
                            # Multi-option radio/checkbox with specific value
                            correct_selector = self._generate_enhanced_selector(question, option_value)
                    else:
                        correct_selector = self._generate_enhanced_selector(question)

                    print(
                        f"DEBUG: _create_answer_data - Generated correct selector: '{correct_selector}' for option '{matched_option.get('text', '')}'")

                    # 🚀 FIXED: Different handling for select vs autocomplete fields
                    if field_type == "autocomplete":
                        # For autocomplete fields, use the readable text as the value
                        select_value = matched_option.get("text", matched_option.get("value", ""))
                        print(
                            f"DEBUG: _create_answer_data - AUTOCOMPLETE field: using text '{select_value}' instead of value '{matched_option.get('value', '')}'")
                    elif field_type == "select":
                        # For regular select fields, use the original value
                        select_value = matched_option.get("value", "")
                        print(
                            f"DEBUG: _create_answer_data - SELECT field: using value '{select_value}' instead of text '{matched_option.get('text', '')}'")
                    else:
                        # For radio/checkbox, keep using the original value
                        select_value = matched_option.get("value", "")

                    # 使用匹配选项的文本作为答案显示
                    return [{
                        "name": matched_option.get("text", matched_option.get("value", "")),
                        "value": select_value,  # Use text for select, value for radio/checkbox
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
                            original_selector = question.get("field_selector", "")

                            # 🚀 ENHANCED: Use enhanced selector generation for default option
                            if field_type == "checkbox" and len(options) == 1:
                                default_selector = self._generate_enhanced_selector(question)
                            else:
                                default_selector = self._generate_enhanced_selector(question, option_value)
                        else:
                            default_selector = self._generate_enhanced_selector(question)

                        # 🚀 FIXED: Different handling for select vs autocomplete fields (default options)
                        if field_type == "autocomplete":
                            default_value = default_option.get("text", default_option.get("value", ""))
                            print(
                                f"DEBUG: _create_answer_data - AUTOCOMPLETE field default: using text '{default_value}' instead of value '{default_option.get('value', '')}'")
                        elif field_type == "select":
                            default_value = default_option.get("value", "")
                            print(
                                f"DEBUG: _create_answer_data - SELECT field default: using value '{default_value}' instead of text '{default_option.get('text', '')}'")
                        else:
                            default_value = default_option.get("value", "")

                        print(
                            f"DEBUG: _create_answer_data - No matching option found for AI answer '{ai_answer_value}', using first option: '{default_option.get('text', '')}' with selector '{default_selector}'")
                        return [{
                            "name": default_option.get("text", default_option.get("value", "")),
                            "value": default_value,  # Use text for select, value for radio/checkbox
                            "check": 1,  # Mark as selected because AI provided an answer
                            "selector": default_selector
                        }]
                    else:
                        # 没有选项，使用AI原始答案作为fallback
                        return [{
                            "name": ai_answer_value,
                            "value": ai_answer_value,
                            "check": 1,  # Mark as selected because AI provided an answer
                            "selector": self._generate_enhanced_selector(question)
                        }]
            else:
                # 没有答案或需要intervention：存储完整选项列表供用户选择
                print(f"DEBUG: _create_answer_data - Option-based field '{question.get('field_name', 'unknown')}' returning option list (needs_intervention={needs_intervention}, has_ai_answer={has_ai_answer})")
                answer_data = []

                for option in options:
                    # 构建选择器
                    if field_type == "autocomplete":
                        selector = self._generate_enhanced_selector(question)
                        # For autocomplete fields, use text as value in option list
                        option_value_to_use = option.get("text", option.get("value", ""))
                    elif field_type == "select":
                        selector = self._generate_enhanced_selector(question)
                        # For regular select fields, use value in option list
                        option_value_to_use = option.get("value", "")
                    else:
                        # 🚀 ENHANCED: Generate enhanced selector for option list
                        option_value = option.get("value", "")
                        option_value_to_use = option_value  # Keep original value for radio/checkbox

                        # Use enhanced selector generation for each option
                        if field_type == "checkbox" and len(options) == 1:
                            selector = self._generate_enhanced_selector(question)
                        else:
                            selector = self._generate_enhanced_selector(question, option_value)

                    answer_data.append({
                        "name": option.get("text", option.get("value", "")),  # 使用选项文本，不是问题文本
                        "value": option_value_to_use,  # Use text for select, value for radio/checkbox
                        "check": 0,  # Not selected - waiting for user input
                        "selector": selector
                    })

                return answer_data
        else:
            # For text inputs, textarea, etc.
            # 🚀 CRITICAL FIX: Consider needs_intervention parameter
            if has_ai_answer and not needs_intervention:
                # AI provided a valid answer and no intervention needed
                return [{
                    "name": ai_answer_value,
                    "value": ai_answer_value,
                    "check": 1,  # Mark as filled because AI provided an answer
                    "selector": self._generate_enhanced_selector(question)
                }]
            else:
                # No answer from AI OR needs intervention - show empty field for user input
                display_value = ai_answer_value if has_ai_answer else ""
                print(f"DEBUG: _create_answer_data - Text field '{question.get('field_name', 'unknown')}' set to check=0 (needs_intervention={needs_intervention}, has_ai_answer={has_ai_answer})")
                return [{
                    "name": display_value,
                    "value": display_value,
                    "check": 0,  # Empty field - waiting for user input
                    "selector": self._generate_enhanced_selector(question)
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
                    current_step.completed_at = datetime.now(datetime.UTC)
                    print(f"[workflow_id:{workflow_id}] DEBUG: Completed current step {current_step_key}")

                    # 2. 激活下一个步骤
                    next_step = self.step_repo.get_step_by_key(workflow_id, next_step_key)
                    if next_step:
                        self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
                        next_step.started_at = datetime.now(datetime.UTC)
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
                # Handle both string response and AIMessage response
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, str):
                    response_content = response
                else:
                    # Try to get content from the response object
                    response_content = str(response)
                
                result = robust_json_parse(response_content)

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
        
        # 🚀 NEW: Cross-page data cache for handling related pages
        self.cross_page_cache = CrossPageDataCache(db_session)

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
    
    def _is_phone_code_field(self, question: Dict[str, Any]) -> bool:
        """Check if a field is a phone/country code field that should have '+' prefix removed"""
        field_name = question.get("field_name", "").lower()
        field_label = question.get("field_label", "").lower()
        field_selector = question.get("field_selector", "").lower()
        
        # Keywords that indicate phone/country code fields
        code_keywords = [
            "international code", "country code", "phone code", "dialing code",
            "area code", "calling code", "telephone code", "mobile code"
        ]
        
        # Check field name, label, and selector for code-related keywords
        for keyword in code_keywords:
            if (keyword in field_name or 
                keyword in field_label or 
                keyword in field_selector):
                return True
        
        # Additional pattern checks
        if "code" in field_name and ("phone" in field_name or "country" in field_name):
            return True
            
        return False

    def _invoke_llm(self, messages: List, workflow_id: str = None):
        """Invoke LLM (workflow_id kept for logging purposes only)"""
        # workflow_id is kept for logging but not used in LLM call
        used_workflow_id = workflow_id or self._current_workflow_id
        if used_workflow_id:
            print(f"[workflow_id:{used_workflow_id}] DEBUG: Invoking LLM")
        return self.llm.invoke(messages)

    def _create_workflow(self) -> StateGraph:
        """Create the streamlined LangGraph workflow with semantic analysis support"""
        workflow = StateGraph(FormAnalysisState)

        # Add nodes - 🚀 STREAMLINED: Removed dependency analyzer, using semantic analysis instead
        workflow.add_node("html_parser", self._html_parser_node)
        workflow.add_node("field_detector", self._field_detector_node)
        workflow.add_node("question_generator", self._question_generator_node)
        # 🚀 NEW: AI semantic analysis of questions to identify logical relationships
        workflow.add_node("semantic_question_analyzer", self._semantic_question_analyzer_node)
        workflow.add_node("profile_retriever", self._profile_retriever_node)
        # 🚀 ASYNC: Use async version for AI-intensive nodes
        workflow.add_node("ai_answerer", self._ai_answerer_node_async)  # Use async version
        # 🚀 NEW: Semantic filtering based on AI answers
        workflow.add_node("semantic_question_filter", self._semantic_question_filter_node)
        # 🚀 NEW: Conditional field analyzer for answer-based group activation
        workflow.add_node("conditional_field_analyzer", self._conditional_field_analyzer_node)
        workflow.add_node("qa_merger", self._qa_merger_node)  # New node for merging Q&A
        workflow.add_node("action_generator", self._action_generator_node)
        # 🚀 ASYNC: Use async version for LLM action generation
        workflow.add_node("llm_action_generator", self._llm_action_generator_node_async)  # Use async version
        workflow.add_node("result_saver", self._result_saver_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Set entry point
        workflow.set_entry_point("html_parser")

        # Add edges - 🚀 STREAMLINED: Direct flow without dependency analysis
        workflow.add_edge("html_parser", "field_detector")
        workflow.add_edge("field_detector", "question_generator")  # 🚀 DIRECT: Skip dependency analysis
        # 🚀 NEW: Add semantic analysis after question generation
        workflow.add_edge("question_generator", "semantic_question_analyzer")
        workflow.add_edge("semantic_question_analyzer", "profile_retriever")
        workflow.add_edge("profile_retriever", "ai_answerer")
        # 🚀 NEW: Add semantic filtering after AI answering
        workflow.add_edge("ai_answerer", "semantic_question_filter")
        workflow.add_edge("semantic_question_filter", "conditional_field_analyzer")
        workflow.add_edge("conditional_field_analyzer", "qa_merger")  # New edge to Q&A merger
        workflow.add_edge("qa_merger", "action_generator")  # 🚀 Updated: Direct to action generator
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

    def process_form(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any],
                     profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process form using LangGraph workflow"""
        try:
            # Reset details expansion tracking for new form
            self._expanded_details = set()
            
            print(f"DEBUG: process_form - Starting with workflow_id: {workflow_id}, step_key: {step_key}")
            print(f"DEBUG: process_form - HTML length: {len(form_html)}")
            print(f"DEBUG: process_form - Profile data keys: {list(profile_data.keys()) if profile_data else 'None'}")
            print(
                f"DEBUG: process_form - Profile dummy data keys: {list(profile_dummy_data.keys()) if profile_dummy_data else 'None'}")

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
                consistency_issues=None,   # 🚀 NEW: Initialize consistency issues
                conditional_context=None,  # 🚀 NEW: Initialize conditional context
                # 🚀 NEW: Initialize conditional field analysis state
                field_selections=None,
                active_field_groups=None,
                conditionally_filtered_questions=None,
                # 🚀 NEW: Initialize semantic question analysis state
                semantic_question_groups=None,
                question_semantic_analysis=None,
                semantically_filtered_questions=None,
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

            # 🚀 CRITICAL FIX: Prioritize precise form_actions over LLM-generated actions
            precise_actions = result.get("form_actions", [])
            llm_actions = result.get("llm_generated_actions", [])
            final_actions = precise_actions if precise_actions else llm_actions

            print(
                f"DEBUG: process_form - Using {'precise' if precise_actions else 'LLM'} actions: {len(final_actions)} total")

            # Return successful result with merged Q&A data
            return {
                "success": True,
                "data": result.get("merged_qa_data", []),  # 返回合并的问答数据
                "actions": final_actions,  # 🚀 优先使用精确动作
                "messages": result.get("messages", []),
                "processing_metadata": {
                    "fields_detected": len(result.get("detected_fields", [])),
                    "questions_generated": len(result.get("field_questions", [])),
                    "answers_generated": len(result.get("ai_answers", [])),
                    "actions_generated": len(final_actions),
                    "action_source": "precise" if precise_actions else "llm",
                    "precise_actions_count": len(precise_actions),
                    "llm_actions_count": len(llm_actions),
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

            # 🚀 NEW: First pass - detect autocomplete pairs
            autocomplete_pairs = {}
            processed_autocomplete = set()

            if hasattr(form_elements, 'find_all'):
                all_elements = form_elements.find_all(["input", "select", "textarea"])

                # Detect autocomplete UI patterns
                for element in all_elements:
                    element_id = element.get("id", "")
                    element_name = element.get("name", "")

                    # Look for autocomplete pattern: input with _ui suffix + corresponding select
                    if (element.name == "input" and
                            element_id.endswith("_ui") and
                            element.get("class") and "ui-autocomplete-input" in str(element.get("class"))):

                        # This is an autocomplete UI input
                        base_id = element_id[:-3]  # Remove "_ui" suffix
                        base_name = element_name.replace("_ui", "") if element_name.endswith("_ui") else ""

                        # Look for corresponding hidden select
                        hidden_select = None
                        for other_elem in all_elements:
                            if (other_elem.name == "select" and
                                    (other_elem.get("id") == base_id or
                                     (base_name and other_elem.get("name") == base_name))):
                                hidden_select = other_elem
                                break

                        if hidden_select:
                            print(
                                f"DEBUG: Field Detector - Found autocomplete pair: UI input '{element_id}' + hidden select '{hidden_select.get('id', '')}'")
                            autocomplete_pairs[base_id] = {
                                "ui_input": element,
                                "hidden_select": hidden_select,
                                "base_name": base_name or base_id
                            }

                # Second pass - process all fields with autocomplete awareness
                for element in all_elements:
                    element_id = element.get("id", "")
                    element_name = element.get("name", "")
                    element_type = element.get("type",
                                               "text").lower() if element.name == "input" else element.name.lower()

                    # Skip if this element is part of an autocomplete pair that we've already processed
                    if element_id in processed_autocomplete:
                        continue

                    # Check if this element is part of an autocomplete pair
                    autocomplete_info = None
                    for base_id, pair_info in autocomplete_pairs.items():
                        if (element == pair_info["ui_input"] or
                                element == pair_info["hidden_select"]):
                            autocomplete_info = pair_info
                            # Mark both elements as processed
                            processed_autocomplete.add(pair_info["ui_input"].get("id", ""))
                            processed_autocomplete.add(pair_info["hidden_select"].get("id", ""))
                            break

                    if autocomplete_info:
                        # Process autocomplete pair as a single field
                        field_info = self._extract_autocomplete_field_info(autocomplete_info)
                        if field_info:
                            detected_fields.append(field_info)
                            print(
                                f"DEBUG: Field Detector - Found autocomplete field: {field_info['name']} ({field_info['type']})")
                        continue

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

                    # Process regular field
                    field_info = self.step_analyzer._extract_field_info(element)
                    if field_info:
                        detected_fields.append(field_info)

                        # Add to field group if it's a radio/checkbox
                        if element_type in ["radio", "checkbox"] and element_name:
                            base_name = re.sub(r'\[\d+\]$', '', element_name)
                            if base_name in field_groups:
                                field_groups[base_name].append(field_info)

                        # 🚀 ENHANCEMENT: Group related fields by semantic similarity
                        # Check if this field is semantically related to existing groups
                        if element_name and element_type not in ["radio", "checkbox"]:
                            for group_name in field_groups.keys():
                                # Check if field name is semantically related to group
                                if (element_name.startswith(group_name) and
                                        element_name != group_name and
                                        len(element_name) > len(group_name)):
                                    # This is a related field (e.g., telephoneNumber relates to telephoneNumberPurpose)
                                    print(
                                        f"DEBUG: Field Detector - Found related field {element_name} for group {group_name}")
                                    # Don't add to group, but mark the relationship for later processing

                        print(f"DEBUG: Field Detector - Found field: {field_info['name']} ({field_info['type']})")

            state["detected_fields"] = detected_fields
            state["field_groups_info"] = field_groups  # Store grouping info

            print(f"DEBUG: Field Detector - Found {len(detected_fields)} fields (after deduplication)")
            print(f"DEBUG: Field Detector - Created {len(field_groups)} field groups")

        except Exception as e:
            print(f"DEBUG: Field Detector - Error: {str(e)}")
            state["error_details"] = f"Field detection failed: {str(e)}"

        return state




    





    




    def _analyze_proximity_grouping(self, detected_fields: List[Dict[str, Any]], html_content: str) -> List[Dict[str, Any]]:
        """🚀 ENHANCED: Advanced HTML structure analysis with AI-powered grouping reasoning"""
        
        groups = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 🚀 STEP 1: Analyze semantic HTML containers with enhanced reasoning
        semantic_containers = soup.find_all(['div', 'section', 'article', 'fieldset', 'details'], class_=True)
        
        for container in semantic_containers:
            container_classes = ' '.join(container.get('class', []))
            container_id = container.get('id', '')
            
            # 🚀 ENHANCED: Smart container classification with reasoning
            container_purpose = self._analyze_container_purpose(container, container_classes, container_id)
            
            # Skip pure layout containers but keep semantic ones
            if container_purpose['type'] == 'layout':
                continue
            
            # Find fields within this container with enhanced matching
            container_fields = self._find_fields_in_container(container, detected_fields)
            
            # 🚀 REASONING: Only group if we have meaningful field relationships
            if len(container_fields) >= 2:
                group_reasoning = self._analyze_group_coherence(container_fields, container_purpose)
                
                if group_reasoning['is_coherent']:
                    groups.append({
                        "group_type": "proximity",
                        "container_class": container_classes,
                        "container_id": container_id,
                        "container_purpose": container_purpose,
                        "fields": container_fields,
                        "field_count": len(container_fields),
                        "reasoning": group_reasoning['reasoning'],
                        "confidence": group_reasoning['confidence']
                    })
        
        # 🚀 STEP 2: Analyze HTML hierarchy patterns (parent-child relationships)
        hierarchy_groups = self._analyze_html_hierarchy_patterns(detected_fields, html_content)
        groups.extend(hierarchy_groups)
        
        # 🚀 STEP 3: Analyze visual layout patterns (same row, column, etc.)
        layout_groups = self._analyze_visual_layout_patterns(detected_fields, html_content)
        groups.extend(layout_groups)
        
        return groups
    
    def _analyze_container_purpose(self, container, container_classes: str, container_id: str) -> Dict[str, Any]:
        """🚀 NEW: Analyze the semantic purpose of HTML containers"""
        
        # Semantic indicators for different purposes
        semantic_indicators = {
            'personal_info': ['personal', 'name', 'identity', 'bio', 'profile'],
            'contact': ['contact', 'phone', 'email', 'address', 'communication'],
            'travel': ['travel', 'journey', 'trip', 'destination', 'passport', 'visa'],
            'family': ['family', 'spouse', 'parent', 'child', 'relative', 'dependent'],
            'employment': ['work', 'job', 'employment', 'employer', 'occupation', 'career'],
            'financial': ['financial', 'money', 'income', 'salary', 'bank', 'cost'],
            'medical': ['health', 'medical', 'doctor', 'hospital', 'treatment'],
            'education': ['education', 'school', 'university', 'qualification', 'degree'],
            'layout': ['container', 'wrapper', 'layout', 'grid', 'row', 'col', 'flex']
        }
        
        combined_text = f"{container_classes} {container_id}".lower()
        
        # Find the most likely purpose
        best_match = {'type': 'unknown', 'confidence': 0, 'indicators': []}
        
        for purpose, indicators in semantic_indicators.items():
            matches = [indicator for indicator in indicators if indicator in combined_text]
            if matches:
                confidence = len(matches) / len(indicators) * 100
                if confidence > best_match['confidence']:
                    best_match = {
                        'type': purpose,
                        'confidence': confidence,
                        'indicators': matches
                    }
        
        # Check for additional semantic clues from container content
        container_text = container.get_text()[:200].lower()  # First 200 chars
        if container_text:
            for purpose, indicators in semantic_indicators.items():
                content_matches = [indicator for indicator in indicators if indicator in container_text]
                if content_matches and purpose != 'layout':
                    best_match['confidence'] += len(content_matches) * 10
                    best_match['indicators'].extend(content_matches)
        
        return best_match
    
    def _find_fields_in_container(self, container, detected_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🚀 ENHANCED: Find fields within container with better matching"""
        container_fields = []
        
        for field in detected_fields:
            field_name = field.get('field_name', '')
            field_id = field.get('id', '')
            
            # Multiple ways to match fields to containers
            if (container.find(attrs={'name': field_name}) or 
                container.find(attrs={'id': field_id}) or
                (field.get('selector') and container.select(field.get('selector')))):
                container_fields.append(field)
        
        return container_fields
    
    def _analyze_group_coherence(self, fields: List[Dict[str, Any]], container_purpose: Dict[str, Any]) -> Dict[str, Any]:
        """🚀 NEW: Analyze if fields form a coherent logical group"""
        
        field_names = [field.get('field_name', '').lower() for field in fields]
        field_labels = [field.get('field_label', '').lower() for field in fields]
        
        # Check semantic coherence
        coherence_score = 0
        reasoning_parts = []
        
        # 1. Container purpose alignment
        if container_purpose['type'] != 'unknown':
            purpose_keywords = {
                'personal_info': ['name', 'first', 'last', 'middle', 'given', 'family'],
                'contact': ['phone', 'email', 'address', 'contact', 'mobile'],
                'travel': ['passport', 'visa', 'country', 'destination', 'date'],
                'family': ['spouse', 'parent', 'child', 'family', 'relationship'],
                'employment': ['job', 'work', 'employer', 'occupation', 'salary'],
                'financial': ['income', 'salary', 'cost', 'financial', 'money'],
                'medical': ['health', 'medical', 'doctor', 'treatment'],
                'education': ['school', 'university', 'qualification', 'education']
            }
            
            purpose_words = purpose_keywords.get(container_purpose['type'], [])
            field_matches = sum(1 for name in field_names + field_labels 
                              if any(keyword in name for keyword in purpose_words))
            
            if field_matches > 0:
                coherence_score += (field_matches / len(fields)) * 30
                reasoning_parts.append(f"Fields match container purpose '{container_purpose['type']}'")
        
        # 2. Field name similarity patterns
        common_prefixes = self._find_common_field_prefixes(field_names)
        if common_prefixes:
            coherence_score += 25
            reasoning_parts.append(f"Fields share common prefixes: {', '.join(common_prefixes)}")
        
        # 3. Field type consistency
        field_types = [field.get('type', '') for field in fields]
        if len(set(field_types)) == 1 and field_types[0] in ['radio', 'checkbox']:
            coherence_score += 20
            reasoning_parts.append(f"All fields are {field_types[0]} type")
        
        # 4. Semantic relationship detection
        semantic_relationships = self._detect_semantic_relationships(field_names + field_labels)
        if semantic_relationships:
            coherence_score += len(semantic_relationships) * 10
            reasoning_parts.extend(semantic_relationships)
        
        is_coherent = coherence_score >= 40  # Threshold for grouping
        
        return {
            'is_coherent': is_coherent,
            'confidence': min(100, coherence_score),
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else 'Basic proximity grouping'
        }
    
    def _find_common_field_prefixes(self, field_names: List[str]) -> List[str]:
        """Find common prefixes in field names"""
        if len(field_names) < 2:
            return []
        
        prefixes = []
        for i, name1 in enumerate(field_names):
            for name2 in field_names[i+1:]:
                # Find common prefix (at least 3 characters)
                common = ''
                for j, (c1, c2) in enumerate(zip(name1, name2)):
                    if c1 == c2:
                        common += c1
                    else:
                        break
                
                if len(common) >= 3 and common not in prefixes:
                    prefixes.append(common)
        
        return prefixes
    
    def _detect_semantic_relationships(self, text_list: List[str]) -> List[str]:
        """Detect semantic relationships between field texts"""
        relationships = []
        
        # Common semantic patterns
        patterns = {
            'date_components': ['day', 'month', 'year', 'date'],
            'name_components': ['first', 'last', 'middle', 'given', 'family', 'name'],
            'address_components': ['street', 'city', 'state', 'zip', 'postal', 'country', 'address'],
            'contact_components': ['phone', 'email', 'mobile', 'telephone', 'contact'],
            'boolean_pairs': ['yes', 'no', 'true', 'false', 'have', 'not']
        }
        
        combined_text = ' '.join(text_list).lower()
        
        for pattern_name, keywords in patterns.items():
            matches = [keyword for keyword in keywords if keyword in combined_text]
            if len(matches) >= 2:
                relationships.append(f"Contains {pattern_name.replace('_', ' ')}: {', '.join(matches)}")
        
        return relationships
    
    def _analyze_html_hierarchy_patterns(self, detected_fields: List[Dict[str, Any]], html_content: str) -> List[Dict[str, Any]]:
        """🚀 NEW: Analyze HTML hierarchy for logical grouping"""
        groups = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find parent elements that contain multiple fields
        for field in detected_fields:
            field_element = None
            field_selector = field.get('selector', '')
            
            if field_selector:
                try:
                    elements = soup.select(field_selector)
                    field_element = elements[0] if elements else None
                except:
                    continue
            
            if field_element:
                # Find parent containers
                parent = field_element.parent
                while parent and parent.name != 'body':
                    # Count how many fields are siblings under this parent
                    sibling_fields = []
                    for other_field in detected_fields:
                        if other_field != field:
                            other_selector = other_field.get('selector', '')
                            if other_selector:
                                try:
                                    other_elements = soup.select(other_selector)
                                    if other_elements and other_elements[0].parent == parent:
                                        sibling_fields.append(other_field)
                                except:
                                    continue
                    
                    # If we found a meaningful group
                    if len(sibling_fields) >= 1:
                        all_fields = [field] + sibling_fields
                        group_id = f"hierarchy_{parent.get('id', '')}_{parent.get('class', [''])[0]}"
                        
                        # Avoid duplicate groups
                        if not any(g.get('group_id') == group_id for g in groups):
                            groups.append({
                                "group_type": "hierarchy",
                                "group_id": group_id,
                                "parent_tag": parent.name,
                                "parent_class": ' '.join(parent.get('class', [])),
                                "fields": all_fields,
                                "field_count": len(all_fields),
                                "reasoning": f"Fields are siblings under {parent.name} element"
                            })
                        break
                    
                    parent = parent.parent
        
        return groups
    
    def _analyze_visual_layout_patterns(self, detected_fields: List[Dict[str, Any]], html_content: str) -> List[Dict[str, Any]]:
        """🚀 NEW: Analyze visual layout patterns for grouping"""
        groups = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for fields in the same row/column based on CSS classes
        layout_indicators = ['row', 'column', 'flex', 'grid', 'inline', 'horizontal', 'vertical']
        
        layout_containers = soup.find_all(attrs={'class': True})
        
        for container in layout_containers:
            container_classes = ' '.join(container.get('class', [])).lower()
            
            # Check if this looks like a layout container
            if any(indicator in container_classes for indicator in layout_indicators):
                container_fields = self._find_fields_in_container(container, detected_fields)
                
                if len(container_fields) >= 2:
                    layout_type = 'unknown'
                    if 'row' in container_classes or 'horizontal' in container_classes:
                        layout_type = 'horizontal'
                    elif 'column' in container_classes or 'vertical' in container_classes:
                        layout_type = 'vertical'
                    elif 'grid' in container_classes:
                        layout_type = 'grid'
                    elif 'flex' in container_classes:
                        layout_type = 'flex'
                    
                    groups.append({
                        "group_type": "visual_layout",
                        "layout_type": layout_type,
                        "container_class": container_classes,
                        "fields": container_fields,
                        "field_count": len(container_fields),
                        "reasoning": f"Fields arranged in {layout_type} layout pattern"
                    })
        
        return groups

    def _analyze_semantic_relationships(self, detected_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🚀 ENHANCED: Advanced semantic analysis with AI-powered reasoning"""
        
        groups = []
        
        # 🚀 ENHANCED: More comprehensive semantic patterns with reasoning
        semantic_patterns = {
            "personal_identity": {
                "keywords": ["first", "last", "middle", "name", "given", "family", "surname", "title", "gender"],
                "group_name": "Personal Identity",
                "reasoning_weight": 35,
                "must_have": ["name"]  # At least one field must contain this
            },
            "contact_information": {
                "keywords": ["phone", "email", "contact", "mobile", "telephone", "fax", "communication"],
                "group_name": "Contact Information", 
                "reasoning_weight": 30,
                "must_have": ["phone", "email"]
            },
            "address_details": {
                "keywords": ["address", "street", "city", "state", "zip", "postal", "country", "location", "residence"],
                "group_name": "Address Information",
                "reasoning_weight": 40,
                "must_have": ["address", "street", "city"]
            },
            "date_information": {
                "keywords": ["date", "birth", "dob", "day", "month", "year", "when", "time"],
                "group_name": "Date Information",
                "reasoning_weight": 25,
                "must_have": ["date", "day", "month", "year"]
            },
            "employment_details": {
                "keywords": ["job", "work", "employer", "company", "occupation", "salary", "employment", "career"],
                "group_name": "Employment Details",
                "reasoning_weight": 30,
                "must_have": ["job", "work", "employer"]
            },
            "travel_information": {
                "keywords": ["passport", "visa", "travel", "destination", "departure", "arrival", "journey", "trip"],
                "group_name": "Travel Information",
                "reasoning_weight": 35,
                "must_have": ["travel", "passport", "visa"]
            },
            "family_details": {
                "keywords": ["spouse", "parent", "child", "family", "relationship", "dependent", "relative", "partner"],
                "group_name": "Family Information",
                "reasoning_weight": 30,
                "must_have": ["family", "spouse", "parent"]
            },
            "financial_information": {
                "keywords": ["income", "salary", "financial", "money", "cost", "bank", "payment", "funds"],
                "group_name": "Financial Information", 
                "reasoning_weight": 25,
                "must_have": ["income", "salary", "financial"]
            },
            "medical_health": {
                "keywords": ["health", "medical", "doctor", "hospital", "treatment", "condition", "medication"],
                "group_name": "Medical Information",
                "reasoning_weight": 30,
                "must_have": ["health", "medical"]
            },
            "education_qualifications": {
                "keywords": ["education", "school", "university", "qualification", "degree", "study", "academic"],
                "group_name": "Education Information",
                "reasoning_weight": 25,
                "must_have": ["education", "school", "qualification"]
            }
        }
        
        for pattern_name, pattern_config in semantic_patterns.items():
            pattern_analysis = self._analyze_semantic_pattern(detected_fields, pattern_config, pattern_name)
            
            if pattern_analysis['is_valid_group']:
                groups.append({
                    "group_type": "semantic",
                    "pattern": pattern_name,
                    "group_name": pattern_config["group_name"],
                    "fields": pattern_analysis['fields'],
                    "field_count": len(pattern_analysis['fields']),
                    "confidence": pattern_analysis['confidence'],
                    "reasoning": pattern_analysis['reasoning'],
                    "semantic_strength": pattern_analysis['semantic_strength']
                })
        
        # 🚀 NEW: Cross-pattern relationship analysis
        cross_pattern_groups = self._analyze_cross_pattern_relationships(detected_fields, groups)
        groups.extend(cross_pattern_groups)
        
        return groups
    
    def _analyze_semantic_pattern(self, detected_fields: List[Dict[str, Any]], pattern_config: Dict[str, Any], pattern_name: str) -> Dict[str, Any]:
        """🚀 NEW: Analyze semantic pattern with advanced reasoning"""
        
        keywords = pattern_config["keywords"]
        must_have = pattern_config.get("must_have", [])
        reasoning_weight = pattern_config.get("reasoning_weight", 20)
        
        pattern_fields = []
        matched_keywords = []
        reasoning_parts = []
        
        # Analyze each field for pattern matching
        for field in detected_fields:
            field_name = field.get('field_name', '').lower()
            field_label = field.get('field_label', '').lower()
            field_text = f"{field_name} {field_label}"
            
            # Count keyword matches
            field_matches = []
            for keyword in keywords:
                if keyword in field_text:
                    field_matches.append(keyword)
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
            
            # If field has semantic matches, add it
            if field_matches:
                field['_semantic_matches'] = field_matches
                field['_semantic_strength'] = len(field_matches)
                pattern_fields.append(field)
        
        # Calculate semantic strength and confidence
        semantic_strength = 0
        confidence = 0
        
        if pattern_fields:
            # 1. Keyword coverage score
            keyword_coverage = len(matched_keywords) / len(keywords) * 100
            semantic_strength += keyword_coverage * 0.4
            
            # 2. Must-have requirements
            must_have_score = 0
            if must_have:
                must_have_found = sum(1 for keyword in must_have if keyword in matched_keywords)
                must_have_score = (must_have_found / len(must_have)) * 100
                semantic_strength += must_have_score * 0.3
                
                if must_have_found > 0:
                    reasoning_parts.append(f"Contains required keywords: {', '.join([k for k in must_have if k in matched_keywords])}")
            
            # 3. Field density (how many fields match vs total)
            field_density = len(pattern_fields) / len(detected_fields) * 100
            if field_density > 20:  # If more than 20% of fields match this pattern
                semantic_strength += field_density * 0.2
                reasoning_parts.append(f"High field density ({field_density:.1f}%) for {pattern_name}")
            
            # 4. Field relationship strength
            if len(pattern_fields) >= 2:
                relationship_strength = self._calculate_field_relationship_strength(pattern_fields)
                semantic_strength += relationship_strength * 0.1
                if relationship_strength > 50:
                    reasoning_parts.append(f"Strong field relationships detected")
            
            confidence = min(100, semantic_strength)
            
            # Add general reasoning
            reasoning_parts.append(f"Matched {len(matched_keywords)}/{len(keywords)} keywords: {', '.join(matched_keywords[:5])}")
            if len(matched_keywords) > 5:
                reasoning_parts.append("...")
        
        # Determine if this is a valid group
        is_valid_group = (
            len(pattern_fields) >= 2 and 
            confidence >= 40 and 
            (not must_have or any(keyword in matched_keywords for keyword in must_have))
        )
        
        return {
            'is_valid_group': is_valid_group,
            'fields': pattern_fields,
            'confidence': confidence,
            'semantic_strength': semantic_strength,
            'matched_keywords': matched_keywords,
            'reasoning': '; '.join(reasoning_parts) if reasoning_parts else f'Basic {pattern_name} pattern matching'
        }
    
    def _calculate_field_relationship_strength(self, fields: List[Dict[str, Any]]) -> float:
        """Calculate how strongly related fields are to each other"""
        if len(fields) < 2:
            return 0
        
        relationship_score = 0
        total_comparisons = 0
        
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:
                total_comparisons += 1
                
                # Compare field names for common patterns
                name1 = field1.get('field_name', '').lower()
                name2 = field2.get('field_name', '').lower()
                
                # Common prefix/suffix
                if len(name1) > 3 and len(name2) > 3:
                    if name1[:3] == name2[:3] or name1[-3:] == name2[-3:]:
                        relationship_score += 20
                
                # Semantic matches overlap
                matches1 = set(field1.get('_semantic_matches', []))
                matches2 = set(field2.get('_semantic_matches', []))
                overlap = len(matches1.intersection(matches2))
                if overlap > 0:
                    relationship_score += overlap * 15
                
                # Field type consistency
                if field1.get('type') == field2.get('type') and field1.get('type') in ['radio', 'checkbox']:
                    relationship_score += 10
        
        return relationship_score / total_comparisons if total_comparisons > 0 else 0
    
    def _analyze_cross_pattern_relationships(self, detected_fields: List[Dict[str, Any]], existing_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """🚀 NEW: Find relationships between different semantic patterns"""
        cross_groups = []
        
        # Look for fields that might bridge different semantic categories
        bridge_patterns = {
            "personal_contact": {
                "patterns": ["personal_identity", "contact_information"],
                "group_name": "Personal Contact Details",
                "reasoning": "Personal identity and contact information are closely related"
            },
            "employment_financial": {
                "patterns": ["employment_details", "financial_information"], 
                "group_name": "Employment & Financial Details",
                "reasoning": "Employment and financial information are interconnected"
            },
            "travel_identity": {
                "patterns": ["travel_information", "personal_identity"],
                "group_name": "Travel Identity Documents", 
                "reasoning": "Travel documents require personal identity information"
            },
            "family_contact": {
                "patterns": ["family_details", "contact_information"],
                "group_name": "Family Contact Information",
                "reasoning": "Family information often includes contact details"
            }
        }
        
        for bridge_name, bridge_config in bridge_patterns.items():
            required_patterns = bridge_config["patterns"]
            
            # Find groups that match the required patterns
            matching_groups = []
            for group in existing_groups:
                if group.get("pattern") in required_patterns:
                    matching_groups.append(group)
            
            # If we have groups from multiple required patterns
            if len(matching_groups) >= 2:
                # Combine fields from matching groups
                combined_fields = []
                combined_confidence = 0
                
                for group in matching_groups:
                    combined_fields.extend(group["fields"])
                    combined_confidence += group.get("confidence", 0)
                
                # Average confidence
                avg_confidence = combined_confidence / len(matching_groups)
                
                # Only create cross-pattern group if confidence is high enough
                if avg_confidence >= 50:
                    cross_groups.append({
                        "group_type": "cross_semantic",
                        "pattern": bridge_name,
                        "group_name": bridge_config["group_name"],
                        "fields": combined_fields,
                        "field_count": len(combined_fields),
                        "confidence": avg_confidence,
                        "reasoning": bridge_config["reasoning"],
                        "source_patterns": required_patterns
                    })
        
        return cross_groups

    def _extract_autocomplete_field_info(self, autocomplete_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract field information from autocomplete UI pair (input + hidden select)"""
        try:
            ui_input = autocomplete_info["ui_input"]
            hidden_select = autocomplete_info["hidden_select"]
            base_name = autocomplete_info["base_name"]

            print(f"DEBUG: _extract_autocomplete_field_info - Processing autocomplete pair for {base_name}")

            # Use the hidden select for the core field info but UI input for interaction
            field_info = {
                "id": hidden_select.get("id", "") or hidden_select.get("name", ""),
                "name": hidden_select.get("name", ""),
                "type": "autocomplete",  # Special type for autocomplete fields
                "selector": f"#{ui_input.get('id', '')}",  # Use UI input selector for interaction
                "hidden_selector": f"#{hidden_select.get('id', '')}",  # Store hidden select selector
                "required": hidden_select.get("required") is not None or ui_input.get("aria-required") == "true",
                "label": self.step_analyzer._find_field_label(ui_input) or self.step_analyzer._find_field_label(
                    hidden_select),
                "placeholder": ui_input.get("placeholder", ""),
                "value": hidden_select.get("value", ""),
                "options": []
            }

            # Extract options from hidden select - for autocomplete, prioritize text over value
            raw_options = self.step_analyzer._extract_field_options(hidden_select)
            # 🚀 FIXED: For autocomplete fields, modify options to use text as value
            autocomplete_options = []
            for option in raw_options:
                # For autocomplete, use text as the value (readable text like "Turkey")
                autocomplete_options.append({
                    "value": option.get("text", option.get("value", "")),  # Use text as value for autocomplete
                    "text": option.get("text", option.get("value", "")),
                    "original_value": option.get("original_value", option.get("value", ""))
                })
            field_info["options"] = autocomplete_options

            print(
                f"DEBUG: _extract_autocomplete_field_info - Autocomplete field: {field_info['name']} with {len(autocomplete_options)} options")
            print(
                f"DEBUG: _extract_autocomplete_field_info - UI selector: {field_info['selector']}, Hidden selector: {field_info['hidden_selector']}")

            return field_info

        except Exception as e:
            print(f"DEBUG: _extract_autocomplete_field_info - Error: {str(e)}")
            return None

    def _question_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """🚀 ENHANCED: Smart question generator with AI-powered field filtering and grouping optimization"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Question Generator - Starting with AI-powered field filtering")

            # Extract page context for better question generation
            page_context = self.step_analyzer._extract_page_context(state["parsed_form"])
            self.step_analyzer._page_context = page_context

            # 🚀 STEP 1: Use all detected fields (dependency analysis removed)
            detected_fields = state["detected_fields"]
            
            # No longer filtering with dependency analysis - using all detected fields
            filtered_fields = detected_fields
            
            # 🚀 STEP 2: Apply universal conditional filtering (TEMPORARILY DISABLED)
            ai_answers = state.get("ai_answers", [])
            form_html = state.get("form_html", "")
            conditional_context = self._build_conditional_context(detected_fields, ai_answers, form_html)
            # DISABLED: filtered_fields = self._filter_fields_by_conditions(filtered_fields, conditional_context)
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Question Generator - Filtered {len(detected_fields)} fields to {len(filtered_fields)} after decision tree and conditional filtering")

            # 🚀 OPTIMIZATION: Use field groups to generate better questions
            field_groups_info = state.get("field_groups_info", {})
            detected_fields = filtered_fields  # Use filtered fields instead of all detected fields

            questions = []
            processed_fields = set()

            # 🔧 NEW: Create a helper function to get element position in HTML
            def get_question_position_in_html(question):
                """Get the position of a question's field element in the HTML document"""
                try:
                    from bs4 import BeautifulSoup
                    html_content = state["form_html"]
                    soup = BeautifulSoup(html_content, 'html.parser')

                    # Try to find the element by selector
                    field_selector = question.get("field_selector", "")
                    if field_selector:
                        # Handle different selector formats
                        if field_selector.startswith("#"):
                            # ID selector
                            element_id = field_selector[1:]
                            element = soup.find(attrs={"id": element_id})
                        elif field_selector.startswith("[name="):
                            # Name attribute selector
                            name_match = re.search(r'\[name=["\']([^"\']+)["\']', field_selector)
                            if name_match:
                                element_name = name_match.group(1)
                                element = soup.find(attrs={"name": element_name})
                            else:
                                element = None
                        else:
                            # Try CSS selector
                            try:
                                elements = soup.select(field_selector)
                                element = elements[0] if elements else None
                            except:
                                element = None

                        if element:
                            # Get all form elements to find position
                            all_elements = soup.find_all(["input", "select", "textarea"])
                            for i, elem in enumerate(all_elements):
                                if elem == element:
                                    return i
                    return 999  # Default high value for elements that can't be found
                except Exception as e:
                    print(f"DEBUG: Question Generator - Error getting position for question: {str(e)}")
                    return 999

            # Process grouped fields first (but collect them for later sorting)
            grouped_questions = []
            for base_name, group_fields in field_groups_info.items():
                if len(group_fields) > 1:
                    # Generate grouped question for radio/checkbox groups
                    grouped_question = self._generate_grouped_question(group_fields, base_name)
                    grouped_questions.append(grouped_question)

                    # Mark fields as processed
                    for field in group_fields:
                        field_id = field.get("id", "")
                        # Ensure field_id is not empty (fallback to name or selector)
                        if not field_id:
                            field_id = field.get("name", "") or field.get("selector", "")

                        if field_id:
                            processed_fields.add(field_id)
                            print(f"DEBUG: Question Generator - Marked field {field_id} as processed (grouped)")

                    print(
                        f"DEBUG: Question Generator - Generated grouped question for {base_name}: {grouped_question['question']}")
                else:
                    # Single field in group - treat as individual field
                    print(f"DEBUG: Question Generator - Single field in group {base_name}, will process individually")

            # Process remaining individual fields
            individual_questions = []
            for field in detected_fields:
                field_id = field.get("id", "")
                field_name = field.get("name", "")

                # Ensure field_id is not empty (fallback to name or selector)
                if not field_id:
                    field_id = field_name or field.get("selector", "")

                if field_id not in processed_fields:
                    question = self.step_analyzer._generate_field_question(field)
                    if question:  # Only add valid questions
                        individual_questions.append(question)
                        print(
                            f"DEBUG: Question Generator - Generated individual question for {field_name} (ID: {field_id}): {question['question']}")
                    else:
                        print(
                            f"DEBUG: Question Generator - Failed to generate question for {field_name} (ID: {field_id})")
                else:
                    print(
                        f"DEBUG: Question Generator - Skipping field {field_name} (ID: {field_id}) - already processed in group")

            # 🚀 NEW: Combine and sort all questions by HTML position
            all_questions = grouped_questions + individual_questions

            # 🚀 NEW: Apply conditional filtering to generated questions (TEMPORARILY DISABLED)
            # DISABLED: all_questions = self._filter_questions_by_conditions(all_questions, conditional_context)

            # Sort questions by their HTML element position
            try:
                sorted_questions = sorted(all_questions, key=get_question_position_in_html)
                print(f"DEBUG: Question Generator - Sorted {len(all_questions)} questions by HTML position")

                # Debug: Show the sorting order
                for i, question in enumerate(sorted_questions):
                    position = get_question_position_in_html(question)
                    field_name = question.get('field_name', 'unknown')
                    print(
                        f"DEBUG: Question Generator - Position {position}: {field_name} - {question.get('question', '')[:50]}...")

                questions = sorted_questions
            except Exception as e:
                print(f"DEBUG: Question Generator - Error sorting questions by position: {str(e)}")
                # Fallback to unsorted order
                questions = all_questions

            # 🚀 NEW: Store conditional context for downstream nodes
            state["conditional_context"] = conditional_context
            state["field_questions"] = questions
            print(
                f"DEBUG: Question Generator - Generated {len(questions)} questions ({len(field_groups_info)} grouped) after conditional filtering, sorted by HTML position")

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
        """🚀 OPTIMIZED: Generate AI answers for questions with batch processing in single call"""
        try:
            print("DEBUG: AI Answerer - Starting with batch processing optimization")

            answers = []
            dummy_usage = []
            profile_data = state.get("profile_data", {})
            profile_dummy_data = state.get("profile_dummy_data", {})
            field_questions = state["field_questions"]

            if not field_questions:
                print("DEBUG: AI Answerer - No questions to process")
                state["ai_answers"] = []
                state["dummy_data_usage"] = []
                return state

            print(f"DEBUG: AI Answerer - Processing {len(field_questions)} questions in SINGLE batch call")

            # 🚀 OPTIMIZATION: Try batch processing first (much more efficient)
            try:
                batch_answers = self._batch_analyze_fields_sync(field_questions, profile_data, profile_dummy_data)
                
                if len(batch_answers) == len(field_questions):
                    print(f"DEBUG: AI Answerer - ✅ Batch processing successful for all {len(batch_answers)} fields")
                    answers = batch_answers
                else:
                    print(f"DEBUG: AI Answerer - ⚠️ Batch processing partial success: {len(batch_answers)}/{len(field_questions)}")
                    # If batch processing didn't return all answers, fall back to individual processing
                    processed_field_names = {ans.get("field_name") for ans in batch_answers if isinstance(ans, dict)}
                    remaining_questions = [q for q in field_questions if q["field_name"] not in processed_field_names]
                    
                    if remaining_questions:
                        print(f"DEBUG: AI Answerer - ⚠️ {len(remaining_questions)} questions remaining, but individual processing removed")
                        print("DEBUG: AI Answerer - Using batch answers only")
                        # 🚀 REMOVED: Individual processing fallback
                        answers = batch_answers
                    else:
                        answers = batch_answers

            except Exception as batch_error:
                print(f"DEBUG: AI Answerer - ❌ Batch processing failed: {str(batch_error)}")
                print("DEBUG: AI Answerer - ⚠️ Individual processing fallback removed - generating empty answers")
                
                # 🚀 REMOVED: Individual processing fallback
                # Instead of calling single-question methods, generate empty answers
                answers = []
                for question in field_questions:
                    empty_answer = {
                        "question_id": question.get("id", ""),
                        "field_selector": question.get("field_selector", ""),
                        "field_name": question.get("field_name", ""),
                        "answer": "",
                        "confidence": 0,
                        "reasoning": "Batch processing failed, individual processing removed",
                        "needs_intervention": True,
                        "used_dummy_data": False
                    }
                    answers.append(empty_answer)
                    
                print(f"DEBUG: AI Answerer - Generated {len(answers)} empty answers as fallback")

            # Process results and track dummy data usage
            for i, answer in enumerate(answers):
                if i < len(field_questions):  # Safety check
                    # Track dummy data usage (including AI-generated dummy data)
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
                            print(f"DEBUG: AI Answerer - Generated smart dummy data for {field_questions[i]['field_name']}: {answer['answer']} (confidence: {answer['confidence']})")
                        else:
                            print(f"DEBUG: AI Answerer - Used provided dummy data for {field_questions[i]['field_name']}: {answer['answer']}")
                    else:
                        print(f"DEBUG: AI Answerer - Used profile data for {field_questions[i]['field_name']}: confidence={answer['confidence']}")

            state["ai_answers"] = answers
            state["dummy_data_usage"] = dummy_usage
            print(f"DEBUG: AI Answerer - Generated {len(answers)} answers, {len(dummy_usage)} used dummy data")

        except Exception as e:
            print(f"DEBUG: AI Answerer - Error: {str(e)}")
            state["error_details"] = f"AI answer generation failed: {str(e)}"

        return state

    def _semantic_question_analyzer_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """🚀 NEW: AI-powered semantic analysis of questions to identify logical relationships"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic Question Analyzer - Starting AI-driven question analysis")
            
            field_questions = state.get("field_questions", [])
            
            if not field_questions:
                print(f"[workflow_id:{workflow_id}] DEBUG: Semantic Question Analyzer - No questions to analyze")
                state["semantic_question_groups"] = []
                state["question_semantic_analysis"] = {}
                return state
            
            # 🚀 STEP 1: Extract question texts for AI analysis
            question_texts = []
            for i, question in enumerate(field_questions):
                question_data = {
                    "index": i,
                    "question": question.get("question", ""),
                    "field_name": question.get("field_name", ""),
                    "field_type": question.get("field_type", ""),
                    "options": question.get("options", []),
                    "required": question.get("required", False)
                }
                question_texts.append(question_data)
            
            # 🚀 STEP 2: Use AI to analyze semantic relationships
            semantic_analysis = self._analyze_question_semantics_with_ai(question_texts, workflow_id)
            
            # 🚀 STEP 3: Create semantic groups based on AI analysis
            semantic_groups = self._create_semantic_question_groups(
                field_questions, 
                semantic_analysis, 
                workflow_id
            )
            
            # 🚀 STEP 4: Store results
            state["semantic_question_groups"] = semantic_groups
            state["question_semantic_analysis"] = semantic_analysis
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic Question Analyzer - Created {len(semantic_groups)} semantic groups")
            
        except Exception as e:
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic Question Analyzer - Error: {str(e)}")
            state["error_details"] = f"Semantic question analysis failed: {str(e)}"
            
        return state
    
    def _analyze_question_semantics_with_ai(self, questions: List[Dict], workflow_id: str) -> Dict[str, Any]:
        """🚀 NEW: Use AI to analyze semantic relationships between questions"""
        
        # Create prompt for AI semantic analysis
        prompt = f"""
                # Task: 分析表单问题之间的语义关系
                
                你正在分析签证申请表单，识别问题之间的逻辑关系。
                重点识别**互斥问题**和**条件依赖关系**。
                
                ## 待分析问题:
                {json.dumps(questions, indent=2, ensure_ascii=False)}
                
                ## 任务:
                分析每个问题的语义含义并识别:
                
                1. **互斥组**: 不能同时回答的问题
                   - 例如: "你有父母吗?" vs "你父亲的姓名是什么?"
                   - 如果第一个问题回答"否"，第二个问题就变得无关紧要
                
                2. **条件依赖**: 依赖其他问题答案的问题
                   - 例如: "你结婚了吗?" → "你配偶的姓名是什么?"
                   - 只有第一个问题是"是"时，第二个问题才有意义
                
                3. **语义分类**: 按语义含义对问题分组
                   - 个人信息、家庭信息、就业、旅行历史等
                
                ## 输出格式:
                返回JSON对象，结构如下:
                {{
                  "mutually_exclusive_groups": [
                    {{
                      "group_name": "parent_info_logic",
                      "reasoning": "如果用户没有父母详细信息，具体父母问题就无关紧要",
                      "trigger_question": {{"index": 5, "question": "What if I do not have my parents' details"}},
                      "excluded_questions": [
                        {{"index": 6, "question": "What is this person's relationship to you?"}},
                        {{"index": 7, "question": "What is your father's name?"}}
                      ],
                      "exclusion_logic": "if_yes_then_exclude"
                    }}
                  ],
                  "conditional_dependencies": [
                    {{
                      "trigger_question": {{"index": 2, "question": "Are you married?"}},
                      "dependent_questions": [
                        {{"index": 8, "question": "What is your spouse's name?"}},
                        {{"index": 9, "question": "When did you get married?"}}
                      ],
                      "dependency_type": "if_yes_then_include"
                    }}
                  ],
                  "semantic_categories": [
                    {{
                      "category": "personal_identity",
                      "questions": [{{"index": 0}}, {{"index": 1}}]
                    }},
                    {{
                      "category": "family_relationships", 
                      "questions": [{{"index": 5}}, {{"index": 6}}, {{"index": 7}}]
                    }}
                  ]
                }}
                
                专注于理解问题的**逻辑含义**，而不仅仅是HTML结构。
                """

        try:
            messages = [
                {"role": "system", "content": "你是分析表单逻辑和问题语义的专家，特别擅长签证申请表单分析。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self._invoke_llm(messages, workflow_id)
            
            if response and response.content:
                # Parse the AI response
                analysis_result = robust_json_parse(response.content)
                return analysis_result
            else:
                return {"error": "No response from AI"}
                
        except Exception as e:
            print(f"[workflow_id:{workflow_id}] DEBUG: AI semantic analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _create_semantic_question_groups(self, questions: List[Dict], semantic_analysis: Dict, workflow_id: str) -> List[Dict]:
        """🚀 NEW: Create question groups based on AI semantic analysis"""
        
        semantic_groups = []
        
        if semantic_analysis.get("error"):
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic analysis had errors, falling back to basic grouping")
            return semantic_groups
        
        # 🎯 Process mutually exclusive groups
        mutually_exclusive = semantic_analysis.get("mutually_exclusive_groups", [])
        for group in mutually_exclusive:
            trigger_q = group.get("trigger_question", {})
            excluded_qs = group.get("excluded_questions", [])
            
            semantic_groups.append({
                "group_type": "mutually_exclusive",
                "group_name": group.get("group_name", ""),
                "reasoning": group.get("reasoning", ""),
                "trigger_question_index": trigger_q.get("index"),
                "excluded_question_indices": [q.get("index") for q in excluded_qs],
                "exclusion_logic": group.get("exclusion_logic", "if_yes_then_exclude"),
                "questions_involved": len(excluded_qs) + 1
            })
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Created mutually exclusive group: {group.get('group_name')}")
        
        # 🎯 Process conditional dependencies
        conditional_deps = semantic_analysis.get("conditional_dependencies", [])
        for dependency in conditional_deps:
            trigger_q = dependency.get("trigger_question", {})
            dependent_qs = dependency.get("dependent_questions", [])
            
            semantic_groups.append({
                "group_type": "conditional_dependency",
                "trigger_question_index": trigger_q.get("index"),
                "dependent_question_indices": [q.get("index") for q in dependent_qs],
                "dependency_type": dependency.get("dependency_type", "if_yes_then_include"),
                "questions_involved": len(dependent_qs) + 1
            })
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Created conditional dependency group")
        
        # 🎯 Process semantic categories
        semantic_categories = semantic_analysis.get("semantic_categories", [])
        for category in semantic_categories:
            category_questions = category.get("questions", [])
            
            semantic_groups.append({
                "group_type": "semantic_category",
                "category_name": category.get("category", ""),
                "question_indices": [q.get("index") for q in category_questions],
                "questions_involved": len(category_questions)
            })
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Created semantic category: {category.get('category')}")
        
        return semantic_groups
    
    def _semantic_question_filter_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """🚀 ENHANCED: Apply semantic-based question filtering with explicit pattern fallbacks"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic Question Filter - Starting")
            
            semantic_groups = state.get("semantic_question_groups", [])
            ai_answers = state.get("ai_answers", [])
            field_questions = state.get("field_questions", [])
            
            if not ai_answers:
                print(f"[workflow_id:{workflow_id}] DEBUG: No AI answers for semantic filtering")
                state["semantically_filtered_questions"] = field_questions
                return state
            
            # Create answer lookup
            answer_lookup = {}
            for answer in ai_answers:
                field_name = answer.get("field_name", "")
                answer_text = answer.get("answer", "").lower().strip()
                answer_lookup[field_name] = answer_text
            
            print(f"[workflow_id:{workflow_id}] DEBUG: AI Answers for semantic filtering: {answer_lookup}")
            
            # Apply semantic filtering rules
            excluded_question_indices = set()
            
            # 🚀 STEP 1: Process AI-generated semantic groups (if available)
            if semantic_groups:
                print(f"[workflow_id:{workflow_id}] DEBUG: Processing {len(semantic_groups)} AI-generated semantic groups")
                
                for group in semantic_groups:
                    if group["group_type"] == "mutually_exclusive":
                        trigger_idx = group.get("trigger_question_index")
                        excluded_indices = group.get("excluded_question_indices", [])
                        exclusion_logic = group.get("exclusion_logic", "if_yes_then_exclude")
                        
                        if trigger_idx is not None and trigger_idx < len(field_questions):
                            trigger_question = field_questions[trigger_idx]
                            trigger_field_name = trigger_question.get("field_name", "")
                            trigger_answer = answer_lookup.get(trigger_field_name, "")
                            
                            print(f"[workflow_id:{workflow_id}] DEBUG: Checking trigger - Field: {trigger_field_name}, Answer: {trigger_answer}")
                            
                            # Check if exclusion condition is met
                            should_exclude = False
                            if exclusion_logic == "if_yes_then_exclude":
                                should_exclude = any(word in trigger_answer for word in ["yes", "true", "是", "有", "don't have", "没有"])
                            elif exclusion_logic == "if_no_then_exclude":
                                should_exclude = any(word in trigger_answer for word in ["no", "false", "否", "don't", "not", "没有"])
                            
                            if should_exclude:
                                excluded_question_indices.update(excluded_indices)
                                print(f"[workflow_id:{workflow_id}] DEBUG: ✅ Excluded {len(excluded_indices)} questions due to trigger: {trigger_answer}")
                            else:
                                print(f"[workflow_id:{workflow_id}] DEBUG: ➡️ No exclusion triggered for: {trigger_answer}")
            
            # 🚀 STEP 2: Apply explicit pattern-based fallback filtering (ENHANCED)
            print(f"[workflow_id:{workflow_id}] DEBUG: Applying explicit pattern-based filtering as fallback")
            
            # Find trigger field for parent exclusion
            parent_trigger_found = False
            for field_name, answer in answer_lookup.items():
                if "parentIsUnknown" in field_name or "parent" in field_name.lower() and "unknown" in field_name.lower():
                    if answer == "true" or "don't have" in answer or "do not have" in answer:
                        print(f"[workflow_id:{workflow_id}] DEBUG: Found parent exclusion trigger: {field_name} = {answer}")
                        parent_trigger_found = True
                        
                        # Exclude all parent-related questions
                        pattern_excluded_count = 0
                        for i, question in enumerate(field_questions):
                            question_field_name = question.get("field_name", "")
                            
                            # Check if this is a parent-related field that should be excluded
                            parent_field_patterns = [
                                "parent.givenName", "parent.familyName", "parent.firstName", "parent.lastName",
                                "parent.dateOfBirth", "parent.dob", "parent.birthday",
                                "parent.nationality", "parent.country", "parent.address",
                                "parent.relationshipRef", "parent.relationship",
                                "parent.hadAlwaysSameNationality", "parent.nationalityAtApplicantsBirthRef"
                            ]
                            
                            for pattern in parent_field_patterns:
                                if pattern.lower() in question_field_name.lower():
                                    if i not in excluded_question_indices:
                                        excluded_question_indices.add(i)
                                        pattern_excluded_count += 1
                                        print(f"[workflow_id:{workflow_id}] DEBUG: Pattern excluded question {i}: {question_field_name}")
                                    break
                        
                        print(f"[workflow_id:{workflow_id}] DEBUG: ✅ Parent exclusion pattern excluded {pattern_excluded_count} questions")
                        break
            
            if not parent_trigger_found:
                print(f"[workflow_id:{workflow_id}] DEBUG: ➡️ No parent exclusion trigger found")
            
            # Filter out excluded questions
            filtered_questions = []
            for i, question in enumerate(field_questions):
                if i not in excluded_question_indices:
                    filtered_questions.append(question)
                else:
                    print(f"[workflow_id:{workflow_id}] DEBUG: 🗑️ Filtered out question {i}: {question.get('question', '')[:50]}...")
            
            state["semantically_filtered_questions"] = filtered_questions
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic filtering: {len(field_questions)} → {len(filtered_questions)} questions")
            
            # Log summary of exclusions
            if excluded_question_indices:
                print(f"[workflow_id:{workflow_id}] DEBUG: ✅ Successfully excluded {len(excluded_question_indices)} questions using semantic filtering")
            else:
                print(f"[workflow_id:{workflow_id}] DEBUG: ➡️ No questions were excluded by semantic filtering")
            
        except Exception as e:
            print(f"[workflow_id:{workflow_id}] DEBUG: Semantic filtering error: {str(e)}")
            state["semantically_filtered_questions"] = state.get("field_questions", [])
            
        return state

    def _conditional_field_analyzer_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """🚀 NEW: Analyze AI answers to determine which field groups should be activated"""
        try:
            print("DEBUG: Conditional Field Analyzer - Starting answer-based field group analysis")
            
            ai_answers = state.get("ai_answers", [])
            detected_fields = state.get("detected_fields", [])
            
            if not ai_answers:
                print("DEBUG: Conditional Field Analyzer - No AI answers to analyze")
                state["active_field_groups"] = []
                return state
            
            # 🚀 STEP 1: Map AI semantic answers to actual field selections
            field_selections = self._map_semantic_to_field_selections(ai_answers, detected_fields)
            
            # 🚀 STEP 2: Determine which conditional field groups should be activated
            active_groups = self._determine_active_field_groups(field_selections, detected_fields)
            
            # 🚀 STEP 3: Filter field questions based on conditional logic
            filtered_questions = self._apply_conditional_field_filtering(
                state["field_questions"], 
                active_groups, 
                field_selections
            )
            
            # 🚀 STEP 4: Update state with conditional analysis results
            state["field_selections"] = field_selections
            state["active_field_groups"] = active_groups  
            state["conditionally_filtered_questions"] = filtered_questions
            
            print(f"DEBUG: Conditional Field Analyzer - Mapped {len(field_selections)} field selections")
            print(f"DEBUG: Conditional Field Analyzer - Activated {len(active_groups)} field groups")
            
        except Exception as e:
            print(f"DEBUG: Conditional Field Analyzer - Error: {str(e)}")
            state["error_details"] = f"Conditional field analysis failed: {str(e)}"
            
        return state
    
    def _map_semantic_to_field_selections(self, ai_answers: List[Dict], detected_fields: List[Dict]) -> Dict[str, str]:
        """🚀 NEW: Map AI semantic answers to specific field option selections"""
        field_selections = {}
        
        for answer in ai_answers:
            field_name = answer.get("field_name", "")
            raw_answer = answer.get("answer", "")
            reasoning = answer.get("reasoning", "")
            
            # Find the corresponding detected field
            field_info = None
            for field in detected_fields:
                if field.get("field_name") == field_name:
                    field_info = field
                    break
            
            if not field_info:
                continue
                
            field_type = field_info.get("type", "")
            field_options = field_info.get("field_options", [])
            
            # 🎯 For radio/checkbox fields, map semantic answer to specific option
            if field_type in ["radio", "checkbox"] and field_options:
                selected_option = self._match_semantic_answer_to_option(raw_answer, reasoning, field_options)
                if selected_option:
                    field_selections[field_name] = selected_option
                    print(f"DEBUG: Mapped '{raw_answer}' → '{selected_option}' for field '{field_name}'")
            
            # 🎯 For text/select fields, use answer directly
            elif field_type in ["text", "select", "email", "number"]:
                field_selections[field_name] = raw_answer
        
        return field_selections
    
    def _match_semantic_answer_to_option(self, ai_answer: str, reasoning: str, field_options: List[Dict]) -> str:
        """🚀 NEW: Match AI semantic answer to specific field option"""
        ai_text = f"{ai_answer} {reasoning}".lower()
        
        best_match = None
        highest_score = 0
        
        for option in field_options:
            option_text = option.get("text", "").lower()
            option_value = option.get("value", "")
            
            # Calculate semantic similarity score
            score = 0
            
            # Direct keyword matching
            if option_text in ai_text or ai_text in option_text:
                score += 50
            
            # Common semantic patterns for visa forms
            semantic_mappings = {
                "married": ["married", "spouse", "husband", "wife", "wed", "marriage"],
                "single": ["single", "unmarried", "not married", "alone", "bachelor"],
                "divorced": ["divorced", "separated", "former spouse", "ex-wife", "ex-husband"],
                "yes": ["yes", "have", "do have", "possess", "own", "true"],
                "no": ["no", "don't have", "do not have", "none", "without", "false"],
                "employed": ["employed", "working", "job", "work", "employee"],
                "unemployed": ["unemployed", "not working", "jobless", "retired", "student"]
            }
            
            for pattern, keywords in semantic_mappings.items():
                if pattern in option_text:
                    for keyword in keywords:
                        if keyword in ai_text:
                            score += 30
                            break
            
            # Update best match
            if score > highest_score:
                highest_score = score
                best_match = option_value
        
        return best_match if highest_score >= 30 else None
    
    def _determine_active_field_groups(self, field_selections: Dict[str, str], detected_fields: List[Dict]) -> List[Dict]:
        """🚀 ENHANCED: Determine which field groups should be activated/deactivated based on selections"""
        active_groups = []
        excluded_field_names = set()  # Track fields that should be excluded
        
        # 🚀 STEP 1: Handle exclusion patterns first (higher priority)
        exclusion_patterns = {
            "parent_exclusion": {
                "trigger_field_patterns": ["parentisunknown", "parent", "noparent"],
                "exclusion_values": ["true", "yes", "unknown", "no details"],
                "excluded_field_patterns": ["parent.", "father.", "mother.", "guardian."],
                "description": "Exclude parent details when user doesn't have them"
            },
            "spouse_exclusion": {
                "trigger_field_patterns": ["marital", "single", "unmarried"],
                "exclusion_values": ["single", "unmarried", "no", "false"],
                "excluded_field_patterns": ["spouse.", "partner.", "husband.", "wife.", "marriage."],
                "description": "Exclude spouse details when user is single"
            },
            "employment_exclusion": {
                "trigger_field_patterns": ["unemployed", "noemployment", "employment"],
                "exclusion_values": ["unemployed", "no", "false", "student", "retired"],
                "excluded_field_patterns": ["employer.", "salary.", "workplace.", "company."],
                "description": "Exclude employment details when user is unemployed"
            }
        }
        
        print(f"DEBUG: Field Selections for exclusion analysis: {field_selections}")
        
        for exclusion_name, pattern in exclusion_patterns.items():
            exclusion_triggered = False
            trigger_info = None
            
            for field_name, selected_value in field_selections.items():
                field_name_lower = field_name.lower()
                selected_value_lower = selected_value.lower()
                
                # Check if this field triggers an exclusion
                trigger_field_match = any(p in field_name_lower for p in pattern["trigger_field_patterns"])
                exclusion_value_match = any(v in selected_value_lower for v in pattern["exclusion_values"])
                
                if trigger_field_match and exclusion_value_match:
                    exclusion_triggered = True
                    trigger_info = {
                        "trigger_field": field_name,
                        "trigger_value": selected_value,
                        "reasoning": f"Field '{field_name}' = '{selected_value}' triggers {exclusion_name}"
                    }
                    print(f"DEBUG: 🚫 EXCLUSION TRIGGERED - {pattern['description']}: {trigger_info['reasoning']}")
                    break
            
            if exclusion_triggered:
                # Find fields to exclude based on patterns
                excluded_count = 0
                for field in detected_fields:
                    field_name_lower = field.get("field_name", "").lower()
                    if any(p in field_name_lower for p in pattern["excluded_field_patterns"]):
                        excluded_field_names.add(field.get("field_name", ""))
                        excluded_count += 1
                        print(f"DEBUG: 🚫 Excluding field: {field.get('field_name', '')}")
                
                print(f"DEBUG: 🚫 Exclusion '{exclusion_name}' triggered, excluded {excluded_count} fields")
        
        # 🚀 STEP 2: Handle inclusion patterns (lower priority)
        inclusion_patterns = {
            "spouse_info": {
                "trigger_field_patterns": ["marital", "marriage", "relationship"],
                "trigger_values": ["married", "wed", "spouse", "husband", "wife"],
                "activated_field_patterns": ["spouse", "partner", "husband", "wife", "marriage"]
            },
            "parent_info": {
                "trigger_field_patterns": ["parent", "family", "guardian"],  
                "trigger_values": ["have", "yes", "details", "known", "alive", "false"],  # Include "false" for parentIsUnknown:false
                "activated_field_patterns": ["father", "mother", "parent", "guardian"]
            },
            "employment_details": {
                "trigger_field_patterns": ["employment", "work", "job", "occupation"],
                "trigger_values": ["employed", "working", "yes", "have job"],
                "activated_field_patterns": ["employer", "salary", "position", "company", "workplace"]
            },
            "travel_history": {
                "trigger_field_patterns": ["travel", "visited", "been"],
                "trigger_values": ["yes", "have", "visited", "been"],
                "activated_field_patterns": ["country", "visit", "trip", "travel", "destination"]
            }
        }
        
        for group_name, pattern in inclusion_patterns.items():
            group_activated = False
            trigger_info = None
            
            for field_name, selected_value in field_selections.items():
                field_name_lower = field_name.lower()
                selected_value_lower = selected_value.lower()
                
                trigger_field_match = any(p in field_name_lower for p in pattern["trigger_field_patterns"])
                trigger_value_match = any(v in selected_value_lower for v in pattern["trigger_values"])
                
                if trigger_field_match and trigger_value_match:
                    group_activated = True
                    trigger_info = {
                        "trigger_field": field_name,
                        "trigger_value": selected_value,
                        "reasoning": f"Field '{field_name}' = '{selected_value}' activates {group_name}"
                    }
                    print(f"DEBUG: ✅ INCLUSION TRIGGERED - {trigger_info['reasoning']}")
                    break
            
            if group_activated:
                # Find fields that belong to this activated group
                group_fields = []
                for field in detected_fields:
                    field_name_lower = field.get("field_name", "").lower()
                    field_label_lower = field.get("field_label", "").lower()
                    field_text = f"{field_name_lower} {field_label_lower}"
                    
                    # Only include if not already excluded
                    if (field.get("field_name", "") not in excluded_field_names and
                        any(p in field_text for p in pattern["activated_field_patterns"])):
                        group_fields.append(field)
                
                if group_fields:
                    active_groups.append({
                        "group_name": group_name,
                        "group_fields": group_fields,
                        "trigger_info": trigger_info,
                        "field_count": len(group_fields)
                    })
                    print(f"DEBUG: ✅ Activated field group '{group_name}' with {len(group_fields)} fields")
        
        # 🚀 STEP 3: Create summary
        print(f"DEBUG: 📊 SUMMARY - Excluded {len(excluded_field_names)} fields, Activated {len(active_groups)} groups")
        if excluded_field_names:
            print(f"DEBUG: 📊 Excluded fields: {list(excluded_field_names)}")
        
        # Store exclusion info in state for downstream nodes
        for group in active_groups:
            group["excluded_fields"] = list(excluded_field_names)
        
        return active_groups
    
    def _apply_conditional_field_filtering(self, questions: List[Dict], active_groups: List[Dict], field_selections: Dict[str, str]) -> List[Dict]:
        """🚀 ENHANCED: Filter questions based on both activated and excluded field groups"""
        
        # Get list of fields that should be excluded (higher priority)
        excluded_field_names = set()
        if active_groups:
            for group in active_groups:
                excluded_fields = group.get("excluded_fields", [])
                excluded_field_names.update(excluded_fields)
        
        # Get list of fields that should be included
        included_field_names = set()
        
        # Include all trigger fields (fields that caused activation/exclusion)
        for field_name in field_selections.keys():
            included_field_names.add(field_name)
        
        # Include fields from activated groups (only if not excluded)
        for group in active_groups:
            for field in group["group_fields"]:
                field_name = field.get("field_name", "")
                if field_name and field_name not in excluded_field_names:
                    included_field_names.add(field_name)
        
        # Always include basic required fields (not conditional) unless explicitly excluded
        basic_field_patterns = ["name", "email", "phone", "address", "passport", "nationality"]
        for question in questions:
            question_field_name = question.get("field_name", "").lower()
            if (any(pattern in question_field_name for pattern in basic_field_patterns) and 
                question.get("field_name", "") not in excluded_field_names):
                included_field_names.add(question.get("field_name", ""))
        
        # If no groups are active but we have exclusions, include all non-excluded fields
        if not active_groups and excluded_field_names:
            print("DEBUG: No active groups but have exclusions, including all non-excluded fields")
            for question in questions:
                question_field_name = question.get("field_name", "")
                if question_field_name not in excluded_field_names:
                    included_field_names.add(question_field_name)
        elif not active_groups and not excluded_field_names:
            print("DEBUG: No active field groups and no exclusions, keeping all questions")
            return questions
        
        # Filter questions to only include relevant ones
        filtered_questions = []
        excluded_count = 0
        
        for question in questions:
            question_field_name = question.get("field_name", "")
            
            # Exclude if explicitly in excluded list
            if question_field_name in excluded_field_names:
                excluded_count += 1
                print(f"DEBUG: 🚫 Excluded field '{question_field_name}' (in exclusion list)")
                continue
            
            # Include if it's in our included set or if we're in inclusion mode
            if question_field_name in included_field_names:
                filtered_questions.append(question)
            else:
                excluded_count += 1
                print(f"DEBUG: 🚫 Filtered out question for field '{question_field_name}' (not in active groups)")
        
        print(f"DEBUG: 📊 Conditional filtering kept {len(filtered_questions)} questions, excluded {excluded_count}")
        if excluded_field_names:
            print(f"DEBUG: 📊 Explicitly excluded {len(excluded_field_names)} fields: {list(excluded_field_names)}")
        
        return filtered_questions

    def _batch_analyze_fields_sync(self, questions: List[Dict[str, Any]], profile_data: Dict[str, Any],
                                   profile_dummy_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """🚀 NEW: Synchronous batch LLM call - analyze multiple fields in single request"""
        try:
            if not questions:
                return []

            print(f"DEBUG: Batch Analysis Sync - Processing {len(questions)} fields in single LLM call")

            # Create batch analysis prompt with all questions
            fields_data = []
            for i, question in enumerate(questions):
                field_data = {
                    "index": i,
                    "field_name": question['field_name'],
                    "field_type": question['field_type'],
                    "field_label": question['field_label'],
                    "question": question['question'],
                    "required": question['required'],
                    "selector": question['field_selector'],
                    "options": question.get('options', [])  # Include options for radio/checkbox fields
                }
                fields_data.append(field_data)

            # 🚀 OPTIMIZED: Simplified prompt to reduce token usage but maintain accuracy
            prompt = f"""
# Task Description
You are the backend of a Google plugin, analyzing HTML pages sent from the frontend, analyzing form items that need to be filled, retrieving provided customer data, filling form content, and returning it to the frontend in a fixed JSON format.

# Important Context - UK Visa Website
⚠️ CRITICAL: This is a UK visa website. For any address-related fields:
- Only addresses within the United Kingdom (England, Scotland, Wales, Northern Ireland) are considered domestic addresses
- Any addresses outside the UK (including EU countries, US, Canada, Australia, etc.) should be treated as international/foreign addresses
- When determining address types or answering location-related questions, apply UK-centric logic

# Fields to analyze ({len(questions)} fields):
{json.dumps(fields_data, indent=1, ensure_ascii=False)}

# User Data:
{json.dumps(profile_data, indent=1, ensure_ascii=False)}

# Fallback Data:
{json.dumps(profile_dummy_data, indent=1, ensure_ascii=False) if profile_dummy_data else "None"}

# Key Instructions:
1. **Semantic Matching** - Understand field meaning, not just field names
2. **Reverse Semantics**: "I do not have X" + "hasX: false" = answer "true" (because user indeed doesn't have X)
3. **Option Matching**: Answer must be exact option text/value, not original data value
4. **Range Matching**: "5 years" + options ["3 years or less", "More than 3 years"] = "More than 3 years"
5. **Country Reasoning**: European countries → "European Economic Area" or "schengen"
6. **Confidence**: 90+ perfect match, 80+ strong semantic match, 60+ reasonable inference

# Character Limit Checking
For each field, check:
- "maxlength" attribute in field data: maxlength="500" means answer must be ≤ 500 characters
- Validation error messages: "maximum 500 characters", "Required field"
- Character count displays: "X characters remaining of Y characters"

# Content Adaptation for Limits
- If data exceeds character limits, prioritize core information and summarize
- For 500 char limit: Include purpose + key dates + essential details only
- Remove redundant phrases, verbose language, unnecessary details
- Maintain factual accuracy while staying within constraints

# Return Format (JSON array, one object per field):
[
    {{
        "index": 0,
        "field_name": "field_name",
        "answer": "value_or_empty_string",
        "confidence": 0-100,
        "reasoning": "brief_explanation",
        "needs_intervention": true/false,
        "data_source_path": "data_path",
        "field_match_type": "exact|semantic|inferred|none",
        "used_dummy_data": true/false
    }},
    ...
]

**IMPORTANT**: Return ONLY the JSON array, no other text. Process ALL {len(questions)} fields.
"""

            # Single LLM call for all fields
            import time
            start_time = time.time()
            
            # Use the existing LLM invoke method
            messages = [{"role": "user", "content": prompt}]
            response = self._invoke_llm(messages, self.workflow_id)
            
            end_time = time.time()
            print(f"DEBUG: Batch Analysis Sync - LLM call completed in {end_time - start_time:.2f}s")

            try:
                # Handle response content
                if hasattr(response, 'content'):
                    response_content = response.content
                elif isinstance(response, str):
                    response_content = response
                else:
                    response_content = str(response)
                
                # Clean and parse JSON response
                response_content = clean_llm_response(response_content)
                results = robust_json_parse(response_content)

                if not isinstance(results, list):
                    raise ValueError("Response is not a list")

                # Validate and format results
                formatted_results = []
                for result in results:
                    if isinstance(result, dict) and "index" in result:
                        index = result["index"]
                        if 0 <= index < len(questions):
                            question = questions[index]
                            
                            # Determine if dummy data was used
                            used_dummy_data = result.get("used_dummy_data", False)
                            if not used_dummy_data:
                                # Check based on data source path or reasoning
                                used_dummy_data = ("contactInformation" in result.get("data_source_path", "") or
                                                 "dummy" in result.get("reasoning", "").lower())

                            # Format the result according to expected structure
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
                                "dummy_data_source": "profile_dummy_data" if used_dummy_data else None,
                                "dummy_data_type": "external" if used_dummy_data else None
                            }
                            formatted_results.append(formatted_result)
                        else:
                            print(f"DEBUG: Batch Analysis Sync - Invalid index {index} for {len(questions)} questions")
                    else:
                        print(f"DEBUG: Batch Analysis Sync - Invalid result format: {result}")

                # Ensure we have results for all questions
                if len(formatted_results) != len(questions):
                    print(f"DEBUG: Batch Analysis Sync - Warning: Expected {len(questions)} results, got {len(formatted_results)}")
                    
                    # Fill missing results with empty answers
                    processed_indices = {r.get("field_name") for r in formatted_results}
                    for i, question in enumerate(questions):
                        if question["field_name"] not in processed_indices:
                            fallback_result = {
                                "question_id": question["id"],
                                "field_selector": question["field_selector"],
                                "field_name": question["field_name"],
                                "answer": "",
                                "confidence": 0,
                                "reasoning": "Batch processing fallback - no result from LLM",
                                "needs_intervention": True,
                                "data_source_path": "",
                                "field_match_type": "none",
                                "used_dummy_data": False,
                                "dummy_data_source": None,
                                "dummy_data_type": None
                            }
                            formatted_results.append(fallback_result)

                print(f"DEBUG: Batch Analysis Sync - Successfully processed {len(formatted_results)} fields")
                
                # Save to cache if available
                if hasattr(self, '_save_to_cache'):
                    cache_data = {
                        "questions": [{"field_name": q["field_name"], "field_type": q["field_type"], "question": q["question"]} for q in questions],
                        "profile_data": profile_data,
                        "profile_dummy_data": profile_dummy_data
                    }
                    cache_key = self._get_cache_key(cache_data)
                    self._save_to_cache(cache_key, formatted_results, "batch")

                return formatted_results

            except Exception as parse_error:
                print(f"DEBUG: Batch Analysis Sync - JSON parsing failed: {str(parse_error)}")
                print(f"DEBUG: Raw response: {response_content[:500]}...")
                raise parse_error

        except Exception as e:
            print(f"DEBUG: Batch Analysis Sync - Error: {str(e)}")
            raise e

    def _qa_merger_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Merge questions and answers into unified data structure using semantically filtered questions"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Q&A Merger - Starting with semantic filtering")

            # 🚀 CRITICAL: Use semantically filtered questions instead of original field_questions
            semantically_filtered_questions = state.get("semantically_filtered_questions", [])
            questions = semantically_filtered_questions if semantically_filtered_questions else state.get("field_questions", [])
            
            print(f"[workflow_id:{workflow_id}] DEBUG: Q&A Merger - Using {'semantically filtered' if semantically_filtered_questions else 'original'} questions")
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
                print(
                    f"DEBUG: Q&A Merger - Processing question group: '{question_text}' with {len(grouped_questions)} fields")

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

                    # 🚀 NEW: Check if this field is conditionally hidden
                    is_conditionally_hidden = self._check_conditional_field_visibility(
                        question, valid_questions, answers, state.get("form_html", "")
                    )
                    
                    # Determine if intervention is needed for this field
                    # Skip intervention for conditionally hidden fields
                    if is_conditionally_hidden:
                        needs_intervention = False
                        print(f"DEBUG: Q&A Merger - Field '{question.get('field_name')}' is conditionally hidden, skipping intervention")
                    else:
                        # 🚀 CRITICAL FIX: Enhanced intervention logic for required fields
                        ai_answer_value = ai_answer.get("answer", "").strip() if ai_answer else ""
                        is_required = question.get("required", False)
                        
                        # Always need intervention if:
                        # 1. No AI answer at all
                        # 2. AI explicitly marked as needing intervention
                        # 3. Low confidence (< 50)
                        # 4. NEW: Required field with empty answer (regardless of confidence)
                        needs_intervention = (not ai_answer or
                                              ai_answer.get("needs_intervention", False) or
                                              ai_answer.get("confidence", 0) < 50 or
                                              (is_required and not ai_answer_value))
                        
                        if is_required and not ai_answer_value:
                            print(f"DEBUG: Q&A Merger - Required field '{question.get('field_name')}' has empty answer, forcing intervention (confidence={ai_answer.get('confidence', 0) if ai_answer else 0})")
                        elif needs_intervention:
                            print(f"DEBUG: Q&A Merger - Field '{question.get('field_name')}' needs intervention: has_answer={bool(ai_answer)}, confidence={ai_answer.get('confidence', 0) if ai_answer else 0}, required={is_required}")

                    all_needs_intervention.append(needs_intervention)
                    all_confidences.append(ai_answer.get("confidence", 0) if ai_answer else 0)
                    all_reasonings.append(ai_answer.get("reasoning", "") if ai_answer else "")

                    # Create answer data for this field
                    # For conditionally hidden fields, create empty data with no intervention needed
                    if is_conditionally_hidden:
                        # Create minimal answer data for hidden field - no actions needed
                        field_answer_data = [{
                            "selector": question.get("field_selector", ""),
                            "value": "",
                            "check": 0,  # Not checked/selected
                            "conditionally_hidden": True
                        }]
                        print(f"DEBUG: Q&A Merger - Created hidden field data for '{question.get('field_name')}'")
                    else:
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

                print(
                    f"DEBUG: Q&A Merger - Question '{question_text}': needs_intervention={overall_needs_intervention}, has_valid_answer={has_valid_answer}, should_interrupt={should_interrupt}")

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

            print(
                f"DEBUG: Q&A Merger - Created {len(merged_data)} merged question groups from {len(questions)} original questions")

        except Exception as e:
            print(f"DEBUG: Q&A Merger - Error: {str(e)}")
            import traceback
            traceback.print_exc()
            state["error_details"] = f"Q&A merging failed: {str(e)}"

        return state

    # =============================================================================
    # 🚀 UNIVERSAL CONDITIONAL DEPENDENCY FRAMEWORK
    # =============================================================================
    
    def _build_conditional_context(self, detected_fields: List[Dict[str, Any]], 
                                   answers: List[Dict[str, Any]], 
                                   form_html: str) -> Dict[str, Any]:
        """Build comprehensive conditional context for field filtering"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(form_html, 'html.parser')
            
            # Extract all conditional rules from HTML
            conditional_rules = self._extract_universal_conditional_rules(form_html, detected_fields)
            
            # Build field state map from current answers
            field_states = self._build_field_state_map(answers)
            
            # Calculate conditional dependencies
            dependency_map = self._calculate_conditional_dependencies(conditional_rules, field_states)
            
            print(f"DEBUG: Conditional Context - Found {len(conditional_rules)} rules, {len(field_states)} states")
            
            return {
                "conditional_rules": conditional_rules,
                "field_states": field_states, 
                "dependency_map": dependency_map,
                "html_soup": soup
            }
            
        except Exception as e:
            print(f"ERROR: Failed to build conditional context: {str(e)}")
            return {
                "conditional_rules": [],
                "field_states": {},
                "dependency_map": {},
                "html_soup": None
            }

    def _extract_universal_conditional_rules(self, html_content: str, 
                                           detected_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conditional rules from HTML in a universal way"""
        rules = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all elements with conditional attributes
        conditional_patterns = [
            "data-toggled-by",
            "data-toggled-by-not", 
            "data-toggle-reverse",
            "data-depends-on",
            "data-show-when",
            "data-hide-when",
            "data-conditional",
            "data-visibility-trigger"
        ]
        
        for pattern in conditional_patterns:
            elements = soup.find_all(attrs={pattern: True})
            
            for element in elements:
                rule = self._parse_conditional_element(element, pattern, detected_fields)
                if rule:
                    rules.append(rule)
        
        # Also check for JavaScript-based conditionals in script tags
        script_rules = self._extract_script_conditionals(html_content, detected_fields)
        rules.extend(script_rules)
        
        # Look for form-specific conditional patterns
        form_rules = self._extract_form_specific_conditionals(html_content, detected_fields)
        rules.extend(form_rules)
        
        print(f"DEBUG: Universal Rules - Extracted {len(rules)} conditional rules")
        return rules

    def _parse_conditional_element(self, element, pattern: str, 
                                 detected_fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse a single conditional element into a rule"""
        try:
            trigger_value = element.get(pattern, "")
            if not trigger_value:
                return None
            
            # Find affected fields within this element
            affected_fields = []
            for field in detected_fields:
                field_name = field.get("field_name", "")
                # Check if field is within this conditional container
                if self._is_field_in_element(field, element):
                    affected_fields.append(field_name)
            
            if not affected_fields:
                return None
            
            # Parse trigger conditions
            conditions = self._parse_trigger_conditions(trigger_value)
            
            # Determine logic type
            is_reverse = (pattern in ["data-toggled-by-not", "data-hide-when"] or 
                         element.get("data-toggle-reverse") == "true")
            
            rule = {
                "type": "conditional_visibility",
                "pattern": pattern,
                "conditions": conditions,
                "affected_fields": affected_fields,
                "reverse_logic": is_reverse,
                "element_tag": element.name,
                "element_id": element.get("id", ""),
                "element_classes": element.get("class", [])
            }
            
            print(f"DEBUG: Rule Parser - Created rule for {len(affected_fields)} fields with {len(conditions)} conditions")
            return rule
            
        except Exception as e:
            print(f"ERROR: Failed to parse conditional element: {str(e)}")
            return None

    def _is_field_in_element(self, field: Dict[str, Any], element) -> bool:
        """Check if a detected field is within a conditional element"""
        field_name = field.get("field_name", "")
        field_selector = field.get("field_selector", "")
        
        # Check by name attribute
        if element.find(attrs={"name": field_name}):
            return True
        
        # Check by id
        if field_selector.startswith("#"):
            field_id = field_selector[1:]
            if element.find(id=field_id):
                return True
        
        # Check by data attributes
        data_attrs = ["data-field", "data-name", "data-field-name"]
        for attr in data_attrs:
            if element.find(attrs={attr: field_name}):
                return True
        
        return False

    def _parse_trigger_conditions(self, trigger_value: str) -> List[Dict[str, Any]]:
        """Parse trigger conditions from conditional attribute value"""
        conditions = []
        
        # Handle multiple conditions separated by comma
        if "," in trigger_value:
            condition_parts = [part.strip() for part in trigger_value.split(",")]
        else:
            condition_parts = [trigger_value.strip()]
        
        for part in condition_parts:
            # Parse condition: "fieldName_expectedValue" or "fieldName:expectedValue" 
            if "_" in part:
                field_name, expected_value = part.rsplit("_", 1)
            elif ":" in part:
                field_name, expected_value = part.rsplit(":", 1)
            elif "=" in part:
                field_name, expected_value = part.rsplit("=", 1)
            else:
                # Default to boolean true
                field_name = part
                expected_value = "true"
            
            conditions.append({
                "field_name": field_name.strip(),
                "expected_value": expected_value.strip().lower(),
                "operator": "equals"  # Could be extended for other operators
            })
        
        return conditions

    def _extract_script_conditionals(self, html_content: str, 
                                   detected_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conditional logic from JavaScript code"""
        rules = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find script tags with conditional logic
        scripts = soup.find_all("script")
        
        for script in scripts:
            if not script.string:
                continue
                
            script_content = script.string
            
            # Look for common JavaScript conditional patterns
            js_patterns = [
                r"if\s*\(\s*([^)]+)\s*\)\s*\{[^}]*\.show\(\)",
                r"if\s*\(\s*([^)]+)\s*\)\s*\{[^}]*\.hide\(\)",
                r"toggle\s*\(\s*([^)]+)\s*\)",
                r"visibility\s*:\s*([^;]+)"
            ]
            
            import re
            for pattern in js_patterns:
                matches = re.finditer(pattern, script_content, re.IGNORECASE)
                for match in matches:
                    # This is a simplified extraction - could be enhanced
                    condition_expr = match.group(1)
                    # Parse JavaScript condition into our rule format
                    # This would need more sophisticated parsing for production use
                    pass
        
        return rules

    def _extract_form_specific_conditionals(self, html_content: str, 
                                          detected_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract form-specific conditional patterns"""
        rules = []
        
        # Re-create BeautifulSoup object from HTML content to avoid serialization issues
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for fieldsets with conditional behavior
        fieldsets = soup.find_all("fieldset")
        for fieldset in fieldsets:
            if fieldset.get("data-conditional") or fieldset.get("data-depends-on"):
                rule = self._parse_conditional_element(fieldset, "data-conditional", detected_fields)
                if rule:
                    rules.append(rule)
        
        # Look for details/summary conditional containers
        details = soup.find_all("details")
        for detail in details:
            # Details elements are naturally conditional (collapsed/expanded)
            affected_fields = []
            for field in detected_fields:
                if self._is_field_in_element(field, detail):
                    affected_fields.append(field.get("field_name", ""))
            
            if affected_fields:
                rules.append({
                    "type": "expandable_section",
                    "pattern": "details",
                    "conditions": [{"field_name": "section_expanded", "expected_value": "true", "operator": "equals"}],
                    "affected_fields": affected_fields,
                    "reverse_logic": False,
                    "element_tag": "details",
                    "element_id": detail.get("id", ""),
                    "auto_expand": detail.get("open") is not None
                })
        
        return rules

    def _build_field_state_map(self, answers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build current field state map from AI answers"""
        field_states = {}
        
        for answer in answers:
            field_name = answer.get("field_name", "")
            answer_value = str(answer.get("answer", "")).lower().strip()
            
            if field_name and answer_value:
                field_states[field_name] = answer_value
                
                # Also store boolean representations
                if answer_value in ["yes", "true", "1", "on", "checked"]:
                    field_states[field_name] = "true"
                elif answer_value in ["no", "false", "0", "off", "unchecked"]:
                    field_states[field_name] = "false"
        
        print(f"DEBUG: Field States - Built state map with {len(field_states)} field states")
        return field_states

    def _calculate_conditional_dependencies(self, conditional_rules: List[Dict[str, Any]], 
                                          field_states: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Calculate which fields should be visible based on current states"""
        dependency_map = {}
        
        # 🚀 FIXED: If no field states (no user answers), don't apply conditional hiding
        # This allows initial form rendering to show all fields
        if not field_states:
            print(f"DEBUG: Dependencies - No field states available, skipping conditional filtering")
            return dependency_map
        
        for rule in conditional_rules:
            affected_fields = rule.get("affected_fields", [])
            conditions = rule.get("conditions", [])
            reverse_logic = rule.get("reverse_logic", False)
            
            # Evaluate all conditions (AND logic by default)
            all_conditions_met = True
            condition_results = []
            has_relevant_answers = False
            
            for condition in conditions:
                field_name = condition.get("field_name", "")
                expected_value = condition.get("expected_value", "")
                actual_value = field_states.get(field_name, "")
                
                # Check if we have a relevant answer for this condition
                if actual_value:
                    has_relevant_answers = True
                
                condition_met = actual_value == expected_value
                condition_results.append({
                    "field": field_name,
                    "expected": expected_value,
                    "actual": actual_value,
                    "met": condition_met
                })
                
                if not condition_met:
                    all_conditions_met = False
            
            # 🚀 FIXED: Only apply hiding if we have relevant answers
            if not has_relevant_answers:
                continue
            
            # Determine visibility based on logic type
            if reverse_logic:
                should_be_visible = not all_conditions_met
            else:
                should_be_visible = all_conditions_met
            
            # Apply to all affected fields
            for field_name in affected_fields:
                dependency_map[field_name] = {
                    "visible": should_be_visible,
                    "rule_type": rule.get("type", "unknown"),
                    "condition_results": condition_results,
                    "reverse_logic": reverse_logic
                }
        
        print(f"DEBUG: Dependencies - Calculated visibility for {len(dependency_map)} fields")
        return dependency_map

    def _should_process_field(self, field: Dict[str, Any], conditional_context: Dict[str, Any]) -> bool:
        """Universal check if a field should be processed based on conditional logic"""
        field_name = field.get("field_name", "")
        
        # If no conditional context, process all fields
        if not conditional_context:
            return True
        
        # 🚀 FIXED: If field_name is empty, default to processing the field
        if not field_name:
            return True
        
        dependency_map = conditional_context.get("dependency_map", {})
        
        # 🚀 FIXED: If field has no conditional rules, it should be processed (default visible)
        if field_name not in dependency_map:
            return True
        
        # Check if field is conditionally visible
        field_dependency = dependency_map[field_name]
        is_visible = field_dependency.get("visible", True)
        
        if not is_visible:
            print(f"DEBUG: Field Filter - Field '{field_name}' is conditionally hidden")
        
        return is_visible

    def _filter_fields_by_conditions(self, fields: List[Dict[str, Any]], 
                                   conditional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter fields list based on conditional visibility"""
        if not conditional_context:
            return fields
        
        filtered_fields = []
        skipped_count = 0
        
        for field in fields:
            if self._should_process_field(field, conditional_context):
                filtered_fields.append(field)
            else:
                skipped_count += 1
        
        print(f"DEBUG: Field Filter - Kept {len(filtered_fields)} fields, skipped {skipped_count} conditionally hidden fields")
        return filtered_fields

    def _filter_questions_by_conditions(self, questions: List[Dict[str, Any]], 
                                      conditional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter questions list based on conditional visibility"""
        if not conditional_context:
            return questions
        
        filtered_questions = []
        skipped_count = 0
        
        for question in questions:
            # Create a field-like object for compatibility
            field_data = {
                "field_name": question.get("field_name", ""),
                "field_selector": question.get("field_selector", "")
            }
            
            if self._should_process_field(field_data, conditional_context):
                filtered_questions.append(question)
            else:
                skipped_count += 1
                print(f"DEBUG: Question Filter - Skipped conditionally hidden question: {question.get('field_name', 'unknown')}")
        
        print(f"DEBUG: Question Filter - Kept {len(filtered_questions)} questions, skipped {skipped_count} conditionally hidden questions")
        return filtered_questions

    def _filter_actions_by_conditions(self, actions: List[Dict[str, Any]], 
                                    conditional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter actions list based on conditional visibility"""
        if not conditional_context:
            return actions
        
        filtered_actions = []
        skipped_count = 0
        
        for action in actions:
            # Extract field name from action selector
            field_name = self._extract_field_name_from_selector(action.get("selector", ""))
            
            field_data = {
                "field_name": field_name,
                "field_selector": action.get("selector", "")
            }
            
            if self._should_process_field(field_data, conditional_context):
                filtered_actions.append(action)
            else:
                skipped_count += 1
                print(f"DEBUG: Action Filter - Skipped conditionally hidden action: {action.get('selector', 'unknown')}")
        
        print(f"DEBUG: Action Filter - Kept {len(filtered_actions)} actions, skipped {skipped_count} conditionally hidden actions")
        return filtered_actions

    def _extract_field_name_from_selector(self, selector: str) -> str:
        """Extract field name from CSS selector"""
        # Handle different selector formats
        if "[name=" in selector:
            # Extract from [name="fieldName"]
            import re
            match = re.search(r'\[name=[\'""]([^\'""]+)[\'""]?\]', selector)
            if match:
                return match.group(1)
        
        if "#" in selector:
            # Extract from ID selector
            return selector.split("#")[-1]
        
        if "data-field=" in selector:
            # Extract from data-field attribute
            import re
            match = re.search(r'data-field=[\'""]([^\'""]+)[\'""]?', selector)
            if match:
                return match.group(1)
        
        return ""

    def _check_conditional_field_visibility(self, question: Dict[str, Any], all_questions: List[Dict[str, Any]], 
                                           answers: List[Dict[str, Any]], form_html: str) -> bool:
        """Check if a field should be hidden due to conditional logic"""
        try:
            field_name = question.get("field_name", "")
            field_selector = question.get("field_selector", "")
            
            # Look for conditional attributes in HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(form_html, 'html.parser')
            
            # Find the field element
            field_element = None
            if field_selector.startswith('#'):
                field_element = soup.find(id=field_selector[1:])
            elif field_name:
                field_element = soup.find(attrs={"name": field_name})
            
            if not field_element:
                return False
                
            # Check if the field or its container has conditional attributes
            conditional_container = field_element.find_parent(attrs={"data-toggled-by": True})
            is_reverse_logic = False
            toggled_by = ""
            
            # Check for standard conditional logic
            if conditional_container:
                toggled_by = conditional_container.get("data-toggled-by", "")
            elif field_element.get("data-toggled-by"):
                conditional_container = field_element
                toggled_by = field_element.get("data-toggled-by", "")
            
            # Check for reverse conditional logic (data-toggled-by-not)
            if not toggled_by:
                conditional_container = field_element.find_parent(attrs={"data-toggled-by-not": True})
                if conditional_container:
                    toggled_by = conditional_container.get("data-toggled-by-not", "")
                    is_reverse_logic = True
                elif field_element.get("data-toggled-by-not"):
                    conditional_container = field_element
                    toggled_by = field_element.get("data-toggled-by-not", "")
                    is_reverse_logic = True
            
            # Check for HTML hidden attribute or CSS display:none
            if not toggled_by:
                if field_element.get("hidden") or field_element.get("aria-hidden") == "true":
                    print(f"DEBUG: Conditional Field - Field '{field_name}' is statically hidden")
                    return True  # Field is hidden
                return False  # No conditional logic found
            
            if not toggled_by:
                return False
                
            print(f"DEBUG: Conditional Field - Field '{field_name}' toggled by '{toggled_by}' (reverse_logic: {is_reverse_logic})")
            
            # Extract trigger field name and expected value
            # Format: "fieldName_expectedValue" (e.g., "warCrimesInvolvement_true")
            # Support multiple conditions: "field1_value1,field2_value2" or single condition
            conditions = []
            if "," in toggled_by:
                # Multiple conditions (AND logic)
                condition_parts = [part.strip() for part in toggled_by.split(",")]
                for part in condition_parts:
                    if "_" in part:
                        field_name = part.rsplit("_", 1)[0]
                        expected_val = part.rsplit("_", 1)[1]
                    else:
                        field_name = part
                        expected_val = "true"
                    conditions.append((field_name, expected_val))
            else:
                # Single condition
                if "_" in toggled_by:
                    trigger_field_name = toggled_by.rsplit("_", 1)[0]
                    expected_value = toggled_by.rsplit("_", 1)[1]
                else:
                    trigger_field_name = toggled_by
                    expected_value = "true"
                conditions.append((trigger_field_name, expected_value))
                
            print(f"DEBUG: Conditional Field - Found {len(conditions)} condition(s) to check")
            
            # Check all conditions (AND logic - all must be satisfied)
            all_conditions_satisfied = True
            condition_results = []
            
            for trigger_field_name, expected_value in conditions:
                print(f"DEBUG: Conditional Field - Checking: '{trigger_field_name}' should be '{expected_value}'")
                
                # Find the trigger field's answer
                trigger_answer = None
                for answer in answers:
                    answer_field_name = answer.get("field_name", "")
                    if trigger_field_name in answer_field_name or answer_field_name in trigger_field_name:
                        trigger_answer = answer
                        break
                        
                if not trigger_answer:
                    print(f"DEBUG: Conditional Field - No trigger answer found for '{trigger_field_name}'")
                    all_conditions_satisfied = False
                    condition_results.append(f"{trigger_field_name}: NOT_FOUND")
                    continue
                    
                # Get the actual value from the trigger field
                actual_value = trigger_answer.get("answer", "").lower()
                expected_value_lower = expected_value.lower()
                
                condition_met = actual_value == expected_value_lower
                condition_results.append(f"{trigger_field_name}: {actual_value}=={expected_value_lower} -> {condition_met}")
                
                if not condition_met:
                    all_conditions_satisfied = False
                    
                print(f"DEBUG: Conditional Field - '{trigger_field_name}': actual='{actual_value}', expected='{expected_value_lower}', met={condition_met}")
            
            print(f"DEBUG: Conditional Field - Condition results: {'; '.join(condition_results)}")
            print(f"DEBUG: Conditional Field - All conditions satisfied: {all_conditions_satisfied}")
            
            # Check if condition is satisfied (all conditions must be satisfied)
            condition_satisfied = all_conditions_satisfied
            
            # Apply reverse logic if needed
            if is_reverse_logic:
                # For reverse logic, we want to hide when conditions ARE satisfied
                should_hide = condition_satisfied
                print(f"DEBUG: Conditional Field - Reverse logic: conditions_satisfied={condition_satisfied}, should_hide={should_hide}")
            else:
                # For normal logic, we hide when conditions are NOT satisfied
                should_hide = not condition_satisfied
                print(f"DEBUG: Conditional Field - Normal logic: conditions_satisfied={condition_satisfied}, should_hide={should_hide}")
            
            if should_hide:
                print(f"DEBUG: Conditional Field - Field '{field_name}' should be hidden (condition not met)")
            else:
                print(f"DEBUG: Conditional Field - Field '{field_name}' should be visible (condition met)")
                
            return should_hide
            
        except Exception as e:
            print(f"DEBUG: Conditional Field - Error checking visibility for '{question.get('field_name')}': {str(e)}")
            return False

    def _action_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node: Generate form actions from merged Q&A data"""
        try:
            workflow_id = state.get("workflow_id", "unknown")
            print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Starting")
            print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Total merged_qa_data items: {len(state.get('merged_qa_data', []))}")

            actions = []
            merged_data = state.get("merged_qa_data", [])
            
            # 🚀 REMOVED: Legacy dependency analysis conditional rules no longer needed
            print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Legacy dependency analysis removed")

            # 🚀 NEW: Apply universal conditional filtering to merged data
            conditional_context = state.get("conditional_context", {})
            if conditional_context:
                # Filter merged data based on conditional context
                original_count = len(merged_data)
                filtered_merged_data = []
                
                for item in merged_data:
                    metadata = item.get("_metadata", {})
                    field_data = {
                        "field_name": metadata.get("field_name", ""),
                        "field_selector": metadata.get("field_selector", "")
                    }
                    
                    if self._should_process_field(field_data, conditional_context):
                        filtered_merged_data.append(item)
                    else:
                        print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Filtered out conditionally hidden field: {metadata.get('field_name', 'unknown')}")
                
                merged_data = filtered_merged_data
                print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Universal conditional filtering: {original_count} -> {len(merged_data)} items")
            else:
                print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - No universal conditional context found")
            
            # 统计各种跳过原因
            skip_stats = {
                "intervention_needed": 0,
                "interrupt_field": 0,
                "no_valid_answer": 0,
                "conditionally_skipped": 0,
                "conditionally_hidden": 0,
                "empty_answer": 0,
                "processed": 0
            }

            for item in merged_data:
                # Extract metadata for action generation
                metadata = item.get("_metadata", {})
                question_data = item.get("question", {})

                # Get answer data for processing
                answer_data = question_data.get("answer", {})
                data_array = answer_data.get("data", [])
                
                # Skip fields that need human intervention
                needs_intervention = metadata.get('needs_intervention', False)
                has_valid_answer = metadata.get('has_valid_answer', False)
                is_interrupt = question_data.get("type") == "interrupt"
                
                # 🚀 NEW: Check if field is conditionally hidden (from answer data)
                # Only consider a field conditionally hidden if ALL items are hidden AND none are checked
                checked_items = [item for item in data_array if item.get("check") == 1]
                hidden_items = [item for item in data_array if item.get("conditionally_hidden", False)]
                is_conditionally_hidden = len(hidden_items) == len(data_array) and len(checked_items) == 0
                
                print(f"DEBUG: Action Generator - Processing {metadata.get('field_name', 'unknown')}")
                print(f"DEBUG: Action Generator - Field type: {metadata.get('field_type', 'unknown')}")
                print(f"DEBUG: Action Generator - Needs intervention: {needs_intervention}")
                print(f"DEBUG: Action Generator - Has valid answer: {has_valid_answer}")
                print(f"DEBUG: Action Generator - Is interrupt: {is_interrupt}")
                print(f"DEBUG: Action Generator - Data array length: {len(data_array)}")
                print(f"DEBUG: Action Generator - Checked items: {len(checked_items)}")
                print(f"DEBUG: Action Generator - Hidden items: {len(hidden_items)}")
                print(f"DEBUG: Action Generator - Is conditionally hidden: {is_conditionally_hidden}")

                # Skip fields that need intervention, are interrupt fields, or are conditionally skipped/hidden
                is_conditionally_skipped = metadata.get('conditional_skip', False)
                if needs_intervention or is_interrupt or not has_valid_answer or is_conditionally_skipped or is_conditionally_hidden:
                    skip_reason = "intervention needed" if needs_intervention else \
                                "interrupt field" if is_interrupt else \
                                "no valid answer" if not has_valid_answer else \
                                "conditionally skipped" if is_conditionally_skipped else \
                                "conditionally hidden"
                    
                    # 更新统计信息
                    if needs_intervention:
                        skip_stats["intervention_needed"] += 1
                    elif is_interrupt:
                        skip_stats["interrupt_field"] += 1
                    elif not has_valid_answer:
                        skip_stats["no_valid_answer"] += 1
                    elif is_conditionally_skipped:
                        skip_stats["conditionally_skipped"] += 1
                    elif is_conditionally_hidden:
                        skip_stats["conditionally_hidden"] += 1
                        
                    print(f"DEBUG: Action Generator - Skipping field {metadata.get('field_name', 'unknown')} - {skip_reason}")
                    print(f"DEBUG: Action Generator - Field details - confidence: {metadata.get('confidence', 'N/A')}, reasoning: {metadata.get('reasoning', 'N/A')[:100]}...")
                    continue


                # Extract answer value from the data array
                answer_value = self._extract_answer_from_data(answer_data)
                print(f"DEBUG: Action Generator - Answer value: '{answer_value}'")

                # Skip if no meaningful answer value
                if not answer_value or answer_value.strip() == "":
                    skip_stats["empty_answer"] += 1
                    print(f"DEBUG: Action Generator - Skipping field {metadata.get('field_name', 'unknown')} - empty answer")
                    print(f"DEBUG: Action Generator - Answer data: {answer_data}")
                    continue
                
                # 字段将被处理
                skip_stats["processed"] += 1

                # 🚀 NEW: Check if field is inside <details> element that needs to be expanded first
                field_selector = metadata.get("field_selector", "")
                details_action = self._check_and_create_details_expand_action(field_selector, state.get("form_html", ""))
                if details_action:
                    actions.append(details_action)
                    print(f"DEBUG: Action Generator - Added details expand action: {details_action}")

                # Generate action for fields with valid answers
                print(
                    f"DEBUG: Action Generator - Processing field {metadata.get('field_name', 'unknown')} - answer: '{answer_value}'")

                # For checkbox and radio fields, we need to generate actions for each checked item
                if metadata.get("field_type") in ["checkbox", "radio"]:
                    data_array = answer_data.get("data", [])
                    precise_actions_generated = False
                    print(f"DEBUG: Action Generator - Processing {metadata.get('field_type')} field with {len(data_array)} options")
                    for data_item in data_array:
                        if data_item.get("check") == 1:
                            # Generate action for this specific checked item
                            action = {
                                "selector": data_item.get("selector", metadata.get("field_selector", "")),
                                "type": "click",
                                "value": data_item.get("value", "")
                            }
                            actions.append(action)
                            precise_actions_generated = True
                            print(f"DEBUG: Action Generator - Generated {metadata.get('field_type')} action: {action}")
                    
                    if precise_actions_generated:
                        print(f"DEBUG: Action Generator - Skipping traditional generation for {metadata.get('field_name')} - precise actions generated")
                        continue  # Skip the traditional action generation only if we generated precise actions
                    else:
                        print(f"DEBUG: Action Generator - No checked items found for {metadata.get('field_name')}, continuing to traditional generation")

                # For non-checkbox fields or checkboxes without checked items, use traditional generation
                # Generate action for input fields and other field types
                if metadata.get("field_type") in ["text", "email", "password", "number", "tel", "url", "date", "time",
                                                  "datetime-local", "textarea", "select", "autocomplete"]:
                    data_array = answer_data.get("data", [])
                    print(f"DEBUG: Action Generator - Processing input field {metadata.get('field_name')} with {len(data_array)} data items")
                    actions_generated_for_field = 0
                    for data_item in data_array:
                        if data_item.get("check") == 1:
                            # 🚀 OPTIMIZATION: For country/location fields, prefer readable text over codes
                            field_name = metadata.get("field_name", "").lower()
                            field_selector = metadata.get("field_selector", "").lower()
                            field_label = metadata.get("field_label", "").lower()
                            action_value = data_item.get("value", "")

                            # For country selection fields, prefer "Turkey" over "TUR", "United Kingdom" over "GBR", etc.
                            # Check field_name, field_selector, and field_label for country-related keywords
                            country_keywords = ["country", "location", "nationality", "birth"]
                            is_country_field = (
                                    any(keyword in field_name for keyword in country_keywords) or
                                    any(keyword in field_selector for keyword in country_keywords) or
                                    any(keyword in field_label for keyword in country_keywords)
                            )

                            if is_country_field:
                                # Check if we have a more readable alternative
                                item_name = data_item.get("name", "")
                                item_value = data_item.get("value", "")

                                # If name is more readable than value (e.g., "Turkey" vs "TUR"), use name
                                if (len(item_name) > len(item_value) and
                                        item_name.isalpha() and
                                        item_value.isupper() and
                                        len(item_value) <= 3):
                                    action_value = item_name
                                    print(
                                        f"DEBUG: Action Generator - Using readable country name '{item_name}' instead of code '{item_value}'")

                            # Generate input action for this field
                            # 🚀 NEW: Determine action type based on field type
                            field_type = metadata.get("field_type", "")
                            if field_type == "autocomplete":
                                # For autocomplete fields, use input action (user types in the UI input)
                                action_type = "input"
                            elif field_type == "select":
                                # For regular select fields, use input action (select from dropdown)
                                action_type = "input"
                            else:
                                # For other field types (text, email, etc.), use input action
                                action_type = "input"

                            action = {
                                "selector": data_item.get("selector", metadata.get("field_selector", "")),
                                "type": action_type,
                                "value": action_value
                            }
                            actions.append(action)
                            actions_generated_for_field += 1
                            print(f"DEBUG: Action Generator - Generated {action['type']} action: {action}")
                            # 🚀 CRITICAL FIX: Process ALL data items, not just the first one
                            # This ensures multiple related fields (e.g., issue date AND expiry date) are all handled
                    print(f"DEBUG: Action Generator - Generated {actions_generated_for_field} actions for input field {metadata.get('field_name')}")
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
                    
            # 🚀 CRITICAL FIX: Always add submit button action at the end
            has_submit = any(
                "submit" in action.get("selector", "").lower() or
                action.get("type") == "submit" or
                ("button" in action.get("selector", "").lower() and "submit" in action.get("selector", "").lower())
                for action in actions
            )

            if not has_submit:
                print("DEBUG: Action Generator - No submit action found, searching for submit button")
                submit_action = self._find_and_create_submit_action(state["form_html"])
                if submit_action:
                    actions.append(submit_action)
                    print(f"DEBUG: Action Generator - Added submit action: {submit_action}")
                else:
                    print("DEBUG: Action Generator - No submit button found in HTML")

            # 🚀 IMPROVED: Sort actions by merged_data order (page question order)
            def get_element_position_in_html(action):
                """Get the position of element based on question order in merged_data"""
                try:
                    selector = action.get("selector", "")
                    
                    # First try to find the action in merged_data order
                    for index, merged_item in enumerate(merged_data):
                        # Check if this action matches any data in this merged_item
                        answer_data = merged_item.get("data", {})
                        data_array = answer_data.get("data", [])
                        
                        # Check if selector matches any data item in this merged question
                        for data_item in data_array:
                            item_selector = data_item.get("selector", "")
                            if item_selector == selector:
                                print(f"DEBUG: Found action {selector} at merged_data position {index}")
                                return index
                    
                    # Fallback: try to extract field name and find in merged_data
                    field_name = ""
                    if selector.startswith("input[") and "name=" in selector:
                        # Extract field name from selector like "input[name='crownDependency']"
                        name_start = selector.find("name=") + 5
                        name_end = selector.find("]", name_start)
                        if name_start > 4 and name_end > name_start:
                            field_name = selector[name_start:name_end].strip('\'"')
                    elif selector.startswith("input[") and "id=" in selector:
                        # Extract field name from selector like "input[id='out-of-crown-dependency']"
                        id_start = selector.find("id=") + 3
                        id_end = selector.find("]", id_start)
                        if id_start > 2 and id_end > id_start:
                            field_name = selector[id_start:id_end].strip('\'"')
                    
                    if field_name:
                        # Find question with matching field name
                        for index, merged_item in enumerate(merged_data):
                            question = merged_item.get("question", {})
                            if (question.get("field_name") == field_name or 
                                field_name in question.get("field_name", "") or
                                field_name in question.get("field_selector", "")):
                                print(f"DEBUG: Found action {selector} by field name match at position {index}")
                                return index
                    
                    print(f"DEBUG: Element {selector} not found in merged_data, using fallback position 9999")
                    return 9999

                except Exception as e:
                    print(f"DEBUG: Error getting element position for {selector}: {str(e)}")
                    return 9999

            # Sort actions by HTML position, keeping submit buttons at the end
            def get_action_sort_key(action):
                selector = action.get("selector", "").lower()
                
                # Submit buttons always go last
                if "submit" in selector or action.get("type") == "submit":
                    return 99999  # Always put submit buttons at the very end
                
                # Details expand actions (summary elements) should go before dependent fields
                # Check if this is a details expand action
                if ("summary" in selector and 
                    ("aria-controls" in selector or action.get("type") == "click")):
                    # Find the lowest position of any field that might depend on this details
                    min_dependent_position = 99999
                    
                    # Extract details ID from aria-controls attribute
                    details_id = None
                    if "aria-controls=" in selector:
                        try:
                            start = selector.find("aria-controls='") + 15
                            end = selector.find("'", start)
                            if start > 14 and end > start:
                                details_id = selector[start:end]
                        except:
                            pass
                    
                    # If we found details ID, look for fields that might be inside this details
                    if details_id:
                        for other_action in actions:
                            other_selector = other_action.get("selector", "")
                            # This is a rough check - in practice, fields inside details would be
                            # positioned after the details element in HTML
                            if other_selector != selector:  # Don't compare with self
                                other_position = get_element_position_in_html(other_action)
                                if other_position < min_dependent_position:
                                    min_dependent_position = other_position
                    
                    # Place details expand action slightly before the first dependent field
                    if min_dependent_position < 99999:
                        return max(0, min_dependent_position - 1)
                    else:
                        return 0  # If no dependent fields found, put at the beginning
                
                # For all other actions, use HTML position
                return get_element_position_in_html(action)

            # Sort actions by HTML element position
            actions.sort(key=get_action_sort_key)

            # 🚀 NEW: Apply final conditional filtering to generated actions
            conditional_context = state.get("conditional_context", {})
            if conditional_context:
                original_action_count = len(actions)
                actions = self._filter_actions_by_conditions(actions, conditional_context)
                print(f"[workflow_id:{workflow_id}] DEBUG: Action Generator - Final conditional filtering: {original_action_count} -> {len(actions)} actions")

            state["form_actions"] = actions
            print(f"DEBUG: Action Generator - Generated {len(actions)} actions total (including submit) after all filtering")
            print("DEBUG: Action Generator - Actions sorted by HTML element position order")
            print(f"DEBUG: Action Generator - Skip statistics: {skip_stats}")
            print(f"DEBUG: Action Generator - Processing success rate: {skip_stats['processed']}/{len(merged_data)} ({skip_stats['processed']/len(merged_data)*100:.1f}% if merged_data else 0)" if merged_data else "No merged data")

            # Debug: Print all generated actions in order
            for i, action in enumerate(actions, 1):
                position = get_action_sort_key(action)
                print(
                    f"DEBUG: Action {i} (HTML Position {position}): {action.get('selector', 'no selector')} -> {action.get('type', 'no type')} ({action.get('value', 'no value')})")

        except Exception as e:
            print(f"DEBUG: Action Generator - Error: {str(e)}")
            import traceback
            traceback.print_exc()
            state["error_details"] = f"Action generation failed: {str(e)}"

        return state

    def _check_and_create_details_expand_action(self, field_selector: str, form_html: str) -> Optional[Dict[str, Any]]:
        """Check if field is inside <details> element and create expand action if needed"""
        try:
            from bs4 import BeautifulSoup
            
            # Track expanded details to avoid duplicates
            if not hasattr(self, '_expanded_details'):
                self._expanded_details = set()
            
            soup = BeautifulSoup(form_html, 'html.parser')
            
            # Find the target field element
            field_element = None
            if field_selector.startswith("#"):
                element_id = field_selector[1:]
                field_element = soup.find(id=element_id)
            elif field_selector.startswith("."):
                class_name = field_selector[1:]
                field_element = soup.find(class_=class_name)
            elif "[" in field_selector and "]" in field_selector:
                # Handle attribute selectors like input[name="parent.relationshipRef"]
                try:
                    tag = field_selector.split("[")[0]
                    attr_part = field_selector.split("[")[1].split("]")[0]
                    if "=" in attr_part:
                        attr_name, attr_value = attr_part.split("=", 1)
                        attr_value = attr_value.strip('"\'')
                        field_element = soup.find(tag, {attr_name: attr_value})
                    else:
                        field_element = soup.find(tag, {attr_part: True})
                except:
                    pass
                    
            if not field_element:
                return None
                
            # Check if field is inside a <details> element
            details_parent = field_element.find_parent('details')
            if not details_parent:
                return None
                
            # Get details element identifier to avoid duplicates
            details_id = details_parent.get('id', '')
            if not details_id:
                # Use summary text as identifier if no id
                summary = details_parent.find('summary')
                if summary:
                    details_id = summary.get_text(strip=True)[:50]  # First 50 chars as ID
                    
            if details_id in self._expanded_details:
                return None  # Already expanded this details element
                
            # Find the summary element
            summary_element = details_parent.find('summary')
            if not summary_element:
                return None
                
            # Mark this details as expanded
            self._expanded_details.add(details_id)
            
            # Create selector for summary element
            summary_selector = None
            if summary_element.get('id'):
                summary_selector = f"#{summary_element['id']}"
            else:
                # Use CSS selector for summary within this details
                if details_parent.get('id'):
                    summary_selector = f"#{details_parent['id']} summary"
                else:
                    # Fallback: use aria-controls attribute if available
                    aria_controls = summary_element.get('aria-controls')
                    if aria_controls:
                        summary_selector = f"summary[aria-controls='{aria_controls}']"
                    else:
                        # Last resort: generic summary selector (risky but better than nothing)
                        summary_selector = "summary"
            
            # Create expand action
            expand_action = {
                "selector": summary_selector,
                "type": "click",
                "value": ""
            }
            
            print(f"DEBUG: Details expand action created - selector: {summary_selector}")
            return expand_action
            
        except Exception as e:
            print(f"DEBUG: Error in _check_and_create_details_expand_action: {str(e)}")
            return None

    def _extract_answer_from_data(self, answer_data: Dict[str, Any]) -> str:
        """Extract answer value from answer data structure"""
        data_array = answer_data.get("data", [])

        # Find all checked items
        checked_items = []
        for item in data_array:
            if item.get("check") == 1:
                # 🚀 OPTIMIZATION: For country/location fields, prefer readable text over codes
                item_name = item.get("name", "")
                item_value = item.get("value", "")

                # Determine which value to use
                value = item_value or item_name

                # For country-like fields, prefer readable names over ISO codes
                if (item_name and item_value and
                        len(item_name) > len(item_value) and
                        item_name.isalpha() and
                        item_value.isupper() and
                        len(item_value) <= 3):
                    value = item_name  # Use "Turkey" instead of "TUR"

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
            print(f"DEBUG: LLM Action Generator - Dummy data fields count: {len(dummy_data_context)}")
            print(f"DEBUG: LLM Action Generator - Profile data structure: {list(state['profile_data'].keys())}")

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

                    # 🚀 CRITICAL FIX: Validate and improve action selectors
                    validated_actions = []
                    for action in actions:
                        validated_action = self._validate_and_improve_action_selector(action, state["form_html"],
                                                                                      dummy_data_context)
                        validated_actions.append(validated_action)

                    actions = validated_actions

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

    def _find_and_create_submit_action(self, form_html: str) -> Optional[Dict[str, Any]]:
        """Find submit button in HTML and create click action for it"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(form_html, 'html.parser')

            # Search for submit buttons in order of priority
            submit_candidates = [
                # Input submit buttons
                soup.find("input", {"type": "submit"}),
                # Button with type submit
                soup.find("button", {"type": "submit"}),
                # Input button with submit-related value
                soup.find("input", {"type": "button"}),
                # Button with submit-related text
                soup.find("button"),
            ]

            # Find the first available submit button
            submit_element = None
            for element in submit_candidates:
                if element:
                    # Check if this is likely a submit button
                    if element.name == "input" and element.get("type") == "submit":
                        submit_element = element
                        break
                    elif element.name == "button" and element.get("type") == "submit":
                        submit_element = element
                        break
                    elif element.name == "input" and element.get("type") == "button":
                        value = element.get("value", "").lower()
                        if any(keyword in value for keyword in ["submit", "continue", "next", "save", "proceed"]):
                            submit_element = element
                            break
                    elif element.name == "button":
                        text = element.get_text(strip=True).lower()
                        if any(keyword in text for keyword in ["submit", "continue", "next", "save", "proceed"]):
                            submit_element = element
                            break

            if submit_element:
                # 🚀 CRITICAL FIX: Create precise selector with proper attribute format
                tag_name = submit_element.name
                element_id = submit_element.get("id")
                element_type = submit_element.get("type")
                element_name = submit_element.get("name")

                if tag_name == "input":
                    # For input elements, use the full attribute format
                    selector_parts = ["input"]

                    # Add ID attribute if available
                    if element_id:
                        selector_parts.append(f"[id='{element_id}']")

                    # Add type attribute
                    if element_type:
                        selector_parts.append(f"[type='{element_type}']")

                    # Add name attribute if available and no ID
                    if element_name and not element_id:
                        selector_parts.append(f"[name='{element_name}']")

                    selector = "".join(selector_parts)

                else:  # button element
                    selector_parts = ["button"]

                    # Add ID attribute if available
                    if element_id:
                        selector_parts.append(f"[id='{element_id}']")

                    # Add type attribute if available
                    if element_type:
                        selector_parts.append(f"[type='{element_type}']")

                    # Add name attribute if available and no ID
                    if element_name and not element_id:
                        selector_parts.append(f"[name='{element_name}']")

                    selector = "".join(selector_parts)

                action = {
                    "selector": selector,
                    "type": "click",
                    "value": None
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

            # 🚀 CRITICAL FIX: Deep clean data to remove circular references
            def clean_circular_references(obj, seen=None, max_depth=10, current_depth=0):
                """Clean circular references from data structure"""
                if seen is None:
                    seen = set()
                
                # Recursion depth protection
                if current_depth > max_depth:
                    return f"<max_depth_reached_at_{max_depth}>"
                
                # Handle different types
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return obj
                
                # Check for circular reference using object id
                obj_id = id(obj)
                if obj_id in seen:
                    return f"<circular_reference_detected>"
                
                seen.add(obj_id)
                
                try:
                    if isinstance(obj, dict):
                        cleaned = {}
                        for k, v in obj.items():
                            # Skip potentially problematic keys that could have circular refs
                            if k in ['parsed_form', 'cross_page_analysis'] and isinstance(v, dict):
                                # Only keep essential data from these fields
                                if k == 'parsed_form':
                                    cleaned[k] = {
                                        "action": v.get("action", ""),
                                        "method": v.get("method", "post")
                                        # Skip html_content to avoid potential BeautifulSoup objects
                                    }
                                else:
                                    cleaned[k] = f"<complex_object_cleaned>"
                            else:
                                cleaned[k] = clean_circular_references(v, seen.copy(), max_depth, current_depth + 1)
                        return cleaned
                    elif isinstance(obj, (list, tuple)):
                        cleaned = []
                        for item in obj:
                            cleaned.append(clean_circular_references(item, seen.copy(), max_depth, current_depth + 1))
                        return cleaned
                    else:
                        # For other objects, try to convert to string representation
                        return str(obj)
                except Exception as e:
                    return f"<serialization_error: {str(e)}>"
                finally:
                    seen.discard(obj_id)

            # Clean the state data before saving
            clean_merged_qa_data = clean_circular_references(state.get("merged_qa_data", []))
            clean_llm_actions = clean_circular_references(state.get("llm_generated_actions", []))
            clean_field_questions = clean_circular_references(state.get("field_questions", []))
            clean_dummy_usage = clean_circular_references(state.get("dummy_data_usage", []))

            # Prepare the main data structure in the expected format
            save_data = {
                "form_data": clean_merged_qa_data,  # 合并的问答数据 - cleaned
                "actions": clean_llm_actions,  # LLM生成的动作 - cleaned
                "questions": clean_field_questions,  # 原始问题数据 - cleaned
                "dummy_data_usage": clean_dummy_usage,  # 虚拟数据使用记录 - cleaned
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "workflow_id": state["workflow_id"],
                    "step_key": state["step_key"],
                    "success": not bool(state.get("error_details")),
                    "field_count": len(state.get("detected_fields", [])),
                    "question_count": len(state.get("field_questions", [])),
                    "answer_count": len(state.get("ai_answers", [])),
                    "action_count": len(state.get("llm_generated_actions", [])),
                    "dummy_data_used_count": len(state.get("dummy_data_usage", [])),
                    "data_cleaned": True,  # Mark that data was cleaned for debugging
                    "circular_ref_protection": True
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
                                "question": str(usage.get("question", ""))[:500],  # Limit length
                                "answer": str(usage.get("answer", ""))[:500],     # Limit length
                                "source": str(usage.get("dummy_data_source", "unknown"))[:100]  # Limit length
                            }
                            simple_records.append(simple_record)

                        # Append both detailed records and simple records to existing array (incremental update)
                        # This ensures both new format and backward compatibility
                        updated_dummy_usage = existing_dummy_usage + simple_records

                        # Update workflow instance
                        workflow_instance.dummy_data_usage = updated_dummy_usage
                        self.db.commit()

                        print(
                            f"DEBUG: Result Saver - Added {len(simple_records)} dummy data usage records to WorkflowInstance")
                        print(f"DEBUG: Result Saver - {len(new_records)} AI-generated dummy data records created")
                        print(f"DEBUG: Result Saver - Total dummy usage records: {len(updated_dummy_usage)}")
                    else:
                        print(
                            f"DEBUG: Result Saver - Warning: WorkflowInstance {state['workflow_id']} not found for dummy data update")

                except Exception as e:
                    print(f"DEBUG: Result Saver - Error updating WorkflowInstance dummy data usage: {str(e)}")
                    # Don't fail the whole operation if this update fails

            # Store saved data in state for reference (cleaned version)
            state["saved_step_data"] = save_data

            print(
                f"DEBUG: Result Saver - Saved data for step {state['step_key']} with {len(save_data['form_data'])} form_data items, {len(save_data['actions'])} actions, and {len(updated_history)} history records")

            # 🚀 NEW: Cache current page data for cross-page relationships
            try:
                # Use cleaned data for caching to avoid circular references
                cache_data = {
                    "form_data": clean_merged_qa_data,
                    "actions": clean_llm_actions,
                    "metadata": save_data["metadata"]
                }
                cache_success = self.cross_page_cache.cache_page_data(
                    state["workflow_id"], 
                    state["step_key"], 
                    cache_data
                )
                if cache_success:
                    print(f"DEBUG: Result Saver - Successfully cached page data for cross-page relationships")
                else:
                    print(f"DEBUG: Result Saver - Failed to cache page data")
            except Exception as cache_error:
                print(f"DEBUG: Result Saver - Error caching page data: {str(cache_error)}")
                # Don't fail the operation if caching fails

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
                    print(
                        f"DEBUG: Error Handler - Created error question for {field.get('name', 'unknown')}: interrupt={should_interrupt}")

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

            print(
                f"DEBUG: Error Handler - Created {len(error_questions)} error questions and saved error data with {len(updated_history)} history records")

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

    async def process_form_async(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any],
                                 profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """🚀 FIXED: True async version using LangGraph workflow with ainvoke + recursion protection"""
        
        # 🚀 CRITICAL FIX: Add recursion protection and retry mechanism
        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                # Reset details expansion tracking for new form
                self._expanded_details = set()
                
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Starting TRUE LangGraph async execution (retry {current_retry + 1}/{max_retries})")
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - HTML length: {len(form_html)}")
                print(
                    f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Profile data keys: {list(profile_data.keys()) if profile_data else 'None'}")
                print(
                    f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Profile dummy data keys: {list(profile_dummy_data.keys()) if profile_dummy_data else 'None'}")

                # Set workflow_id for thread isolation
                self.set_workflow_id(workflow_id)

                # Create initial state with circular reference protection
                def clean_input_data(data):
                    """Clean input data to prevent circular references from the start"""
                    if not isinstance(data, dict):
                        return data
                    
                    cleaned = {}
                    for key, value in data.items():
                        if isinstance(value, dict):
                            # Recursively clean nested dictionaries 
                            cleaned[key] = clean_input_data(value)
                        elif isinstance(value, list):
                            # Clean list items
                            cleaned[key] = [clean_input_data(item) if isinstance(item, dict) else item for item in value]
                        else:
                            cleaned[key] = value
                    return cleaned
                
                # 🚀 CRITICAL FIX: Clean input data before state creation
                clean_profile_data = clean_input_data(profile_data or {})
                clean_profile_dummy_data = clean_input_data(profile_dummy_data or {})
                
                initial_state = FormAnalysisState(
                    workflow_id=workflow_id,
                    step_key=step_key,
                    form_html=form_html,
                    profile_data=clean_profile_data,
                    profile_dummy_data=clean_profile_dummy_data,
                parsed_form=None,
                detected_fields=[],
                field_questions=[],
                ai_answers=[],
                merged_qa_data=[],
                form_actions=[],
                llm_generated_actions=[],
                saved_step_data=None,
                dummy_data_usage=[],
                consistency_issues=None,   # 🚀 NEW: Initialize consistency issues
                conditional_context=None,  # 🚀 NEW: Initialize conditional context
                # 🚀 NEW: Initialize conditional field analysis state
                field_selections=None,
                active_field_groups=None,
                conditionally_filtered_questions=None,
                # 🚀 NEW: Initialize semantic question analysis state
                semantic_question_groups=None,
                question_semantic_analysis=None,
                semantically_filtered_questions=None,
                analysis_complete=False,
                error_details=None,
                messages=[]
            )

                # 🚀 FIXED: Use true LangGraph async execution with proper config
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Running LangGraph workflow with ainvoke")
                
                # Provide config for checkpointer - required for async execution
                import uuid
                thread_id = str(uuid.uuid4())
                config = {
                    "configurable": {
                        "thread_id": thread_id  # Use UUID for thread isolation
                    }
                }
                
                print(f"[workflow_id:{workflow_id}] DEBUG: Using thread_id: {thread_id}")
                result = await self.app.ainvoke(initial_state, config=config)

                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - LangGraph async workflow completed")
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Result keys: {list(result.keys())}")

                # 🚀 SUCCESS PATH: Process successful workflow result
                # Check for errors
                if result.get("error_details"):
                    return {
                        "success": False,
                        "error": result["error_details"],
                        "data": [],
                        "actions": []
                    }

                # 🚀 CRITICAL FIX: Prioritize precise form_actions over LLM-generated actions
                precise_actions = result.get("form_actions", [])
                llm_actions = result.get("llm_generated_actions", [])
                final_actions = precise_actions if precise_actions else llm_actions

                print(
                    f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Using {'precise' if precise_actions else 'LLM'} actions: {len(final_actions)} total")

                # Return successful result with merged Q&A data
                return {
                    "success": True,
                    "data": result.get("merged_qa_data", []),  # 返回合并的问答数据
                    "actions": final_actions,  # 🚀 优先使用精确动作
                    "messages": result.get("messages", []),
                    "processing_metadata": {
                        "fields_detected": len(result.get("detected_fields", [])),
                        "questions_generated": len(result.get("field_questions", [])),
                        "answers_generated": len(result.get("ai_answers", [])),
                        "actions_generated": len(final_actions),
                        "action_source": "precise" if precise_actions else "llm",
                        "precise_actions_count": len(precise_actions),
                        "llm_actions_count": len(llm_actions),
                        "workflow_id": workflow_id,
                        "step_key": step_key
                    }
                }

            except Exception as workflow_error:
                current_retry += 1
                error_message = str(workflow_error)
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Retry {current_retry}/{max_retries} - Error: {error_message}")
                
                # 🚀 CRITICAL: Check for specific recursion/circular reference errors
                if "recursion" in error_message.lower() or "circular" in error_message.lower():
                    print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Detected recursion/circular reference error")
                    # Force a clean slate for the next retry
                    import gc
                    gc.collect()
                    
                    # If this is the last retry, return graceful failure
                    if current_retry >= max_retries:
                        print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Max retries reached for recursion error")
                        return {
                            "success": False,
                            "error": f"LangGraph workflow failed after {max_retries} retries due to recursion/circular reference: {error_message}",
                            "data": [],
                            "actions": []
                        }
                    
                    # Continue to next retry
                    continue
                
                # For other errors, fail fast
                print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - Non-recursion error, failing fast")
                return {
                    "success": False,
                    "error": f"LangGraph workflow failed: {error_message}",
                    "data": [],
                    "actions": []
                }
        
        # If we exit the while loop without returning, all retries failed
        print(f"[workflow_id:{workflow_id}] DEBUG: process_form_async - All retries exhausted")
        return {
            "success": False,
            "error": f"LangGraph workflow failed after {max_retries} retries",
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
            workflow_id = state.get("workflow_id", "unknown")
            step_key = state.get("step_key", "unknown")

            if not field_questions:
                print("DEBUG: AI Answerer Async - No questions to process")
                state["ai_answers"] = []
                state["dummy_data_usage"] = []
                return state

            print(f"DEBUG: AI Answerer Async - Processing {len(field_questions)} questions with advanced optimizations")
            
            # 🚀 NEW: Check for cross-page data before processing
            try:
                # Check if current page has questions about "other/additional" items
                has_cross_page_questions = any(
                    any(keyword in question.get("question", "").lower() for keyword in 
                        ["other", "additional", "another", "more", "还有", "其他", "另外"]) 
                    for question in field_questions
                )
                
                if has_cross_page_questions:
                    print(f"DEBUG: AI Answerer Async - Detected cross-page questions, analyzing previous data")
                    
                    # Analyze address completion for address-related questions
                    if any(keyword in question.get("question", "").lower() for keyword in 
                           ["address", "lived", "residence", "地址", "居住"] for question in field_questions):
                        
                        address_analysis = self.cross_page_cache.analyze_address_completion(
                            workflow_id, step_key, profile_data
                        )
                        
                        print(f"DEBUG: AI Answerer Async - Address analysis: "
                              f"{len(address_analysis.get('filled_addresses', []))} filled, "
                              f"{len(address_analysis.get('remaining_addresses', []))} remaining, "
                              f"has_other: {address_analysis.get('has_other_addresses', False)}")
                        
                        # Store analysis in state for later use
                        state["cross_page_analysis"] = address_analysis
                        
                        # Apply contextual answers for specific questions
                        for question in field_questions:
                            question_text = question.get("question", "").lower()
                            if any(keyword in question_text for keyword in 
                                   ["other address", "lived at any other", "还有", "其他地址"]):
                                
                                # Add contextual information to the question
                                question["cross_page_context"] = {
                                    "has_other_addresses": address_analysis.get("has_other_addresses", False),
                                    "filled_addresses": address_analysis.get("filled_addresses", []),
                                    "remaining_addresses": address_analysis.get("remaining_addresses", []),
                                    "reasoning": f"Based on previous pages: {len(address_analysis.get('filled_addresses', []))} addresses filled, {len(address_analysis.get('remaining_addresses', []))} remaining"
                                }
                                
                                print(f"DEBUG: AI Answerer Async - Added cross-page context to question: {question.get('field_name', 'unknown')}")
                
            except Exception as cross_page_error:
                print(f"DEBUG: AI Answerer Async - Cross-page analysis failed: {str(cross_page_error)}")
                # Continue processing without cross-page data
                pass

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
                        # Get cross-page context for remaining questions
                        cross_page_context = state.get("cross_page_analysis", {})
                        
                        batch_answers = await self._batch_analyze_fields_async(remaining_questions, profile_data,
                                                                               profile_dummy_data, cross_page_context)
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
                    # Get cross-page context for batch processing
                    cross_page_context = state.get("cross_page_analysis", {})
                    
                    batch_answers = await self._batch_analyze_fields_async(field_questions, profile_data,
                                                                           profile_dummy_data, cross_page_context)

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
                            print(f"DEBUG: AI Answerer Async - ⚠️ {len(remaining_questions)} questions remaining, but individual processing removed")
                            print("DEBUG: AI Answerer Async - Using batch answers only")
                            # 🚀 REMOVED: Individual processing fallback
                            answers = batch_answers
                        else:
                            answers = batch_answers

                except Exception as batch_error:
                    print(f"DEBUG: AI Answerer Async - ❌ Batch processing failed: {str(batch_error)}")
                    print("DEBUG: AI Answerer Async - ⚠️ Individual processing fallback removed - generating empty answers")

                    # 🚀 REMOVED: Individual processing fallback
                    # Instead of calling single-question methods, generate empty answers
                    answers = []
                    for question in field_questions:
                        empty_answer = {
                            "question_id": question.get("id", ""),
                            "field_selector": question.get("field_selector", ""),
                            "field_name": question.get("field_name", ""),
                            "answer": "",
                            "confidence": 0,
                            "reasoning": "Batch processing failed, individual processing removed",
                            "needs_intervention": True,
                            "used_dummy_data": False
                        }
                        answers.append(empty_answer)
                        
                    print(f"DEBUG: AI Answerer Async - Generated {len(answers)} empty answers as fallback")

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

# 🚀 REMOVED: _generate_ai_answer_async method - now using batch processing only

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
                                          profile_dummy_data: Dict[str, Any] = None, 
                                          cross_page_context: Dict[str, Any] = None) -> List[
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

            # 🚀 NEW: Enhanced contextual reasoning
            enhanced_reasoning = self._enhanced_contextual_reasoning(questions, profile_data, profile_dummy_data)
            confidence_boost = enhanced_reasoning["confidence_boost"]["overall_boost"]
            
            print(f"DEBUG: Enhanced Reasoning - Confidence boost: {confidence_boost}")
            print(f"DEBUG: Enhanced Reasoning - Confident answers: {enhanced_reasoning['confidence_boost']['confident_answers']}/{enhanced_reasoning['confidence_boost']['total_questions']}")

            # Create batch analysis prompt with enhanced context
            fields_data = []
            for i, question in enumerate(questions):
                reasoning_chain = enhanced_reasoning["reasoning_chains"][i] if i < len(enhanced_reasoning["reasoning_chains"]) else None
                
                field_data = {
                    "index": i,
                    "field_name": question['field_name'],
                    "field_type": question['field_type'],
                    "field_label": question['field_label'],
                    "question": question['question'],
                    "required": question['required'],
                    "selector": question['field_selector'],
                    "options": question.get('options', [])  # Include options for radio/checkbox fields
                }
                
                # Add reasoning chain information if available
                if reasoning_chain and reasoning_chain["final_confidence"] > 0:
                    field_data["pre_reasoning"] = {
                        "confidence": reasoning_chain["final_confidence"],
                        "steps": reasoning_chain["steps"],
                        "decision_path": reasoning_chain["decision_path"]
                    }
                
                fields_data.append(field_data)

            # 🚀 OPTIMIZATION: Simplified prompt to reduce token usage with cross-page context
            prompt = f"""
            Analyze form fields and provide answers based on user data.

            # Fields to analyze:
            {json.dumps(fields_data, indent=1, ensure_ascii=False)}

            # User Data:
            {json.dumps(profile_data, indent=1, ensure_ascii=False)}

            # Dummy Data:
            {json.dumps(profile_dummy_data, indent=1, ensure_ascii=False) if profile_dummy_data else "None"}

            # Cross-Page Context (from previous form pages):
            {json.dumps(cross_page_context, indent=1, ensure_ascii=False) if cross_page_context else "None"}

            # Instructions:
            1. Use semantic matching - understand field meaning, not just names
            2. For negative statements: "I do not have X" + "hasX: false" = answer "true"
            3. For options: answer must be exact option text/value, not original data
            4. For ranges: "5 years" + options ["3 years or less", "More than 3 years"] = "More than 3 years"
            5. For countries: European countries → "European Economic Area" or "schengen"
            6. Confidence: 90+ for perfect match, 80+ for strong semantic match
            7. For phone/country/international CODE fields: ALWAYS remove the "+" prefix if present in the data
              * Field asking for "international code", "country code", "phone code" etc.
              * If data contains "+90", "+1", "+44" etc., return only the digits: "90", "1", "44"
              * Examples: "+90" → "90", "+44" → "44", "+1" → "1"
              * This applies to any field that semantically represents a phone country code
            8. 🚀 NEW: Cross-page context usage:
              * For questions about "other/additional/more" items, use cross-page context
              * If context shows previous addresses/items filled, answer "yes/no" accordingly
              * For "Have you lived at any other addresses?": check filled_addresses vs remaining_addresses
              * High confidence (80+) for cross-page contextual answers with clear previous data
            Response format (JSON array):
            [
              {{
                "index": 0,
                "answer": "value_or_empty",
                "confidence": 0-100,
                "reasoning": "brief_explanation",
                "needs_intervention": true/false,
                "field_match_type": "exact|semantic|inferred|none"
              }}
            ]
            
            **Character Limits**: For each field, check:
            - "maxlength" attribute in field data: maxlength="500" means answer must be ≤ 500 characters
            - Validation error messages: "maximum 500 characters", "Required field"
            - Character count displays: "X characters remaining of Y characters"
            
            **Content Adaptation for Limits**:
            - If data exceeds character limits, prioritize key information and summarize
            - For 500 char limit: Include purpose + key dates + essential details only
            - Remove redundant phrases, verbose language, unnecessary details
            - Maintain factual accuracy while staying within constraints
            
            **Constraint Examples**:
            - Field with maxlength="500" → Answer MUST be ≤ 500 characters
            - Validation showing "maximum 500 characters" → Shorten existing content
            - Required field → Generate appropriate content or flag for intervention

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
                # Handle both string response and AIMessage response
                if hasattr(response, 'content'):
                    response_content = clean_llm_response_array(response.content)
                elif isinstance(response, str):
                    response_content = response
                else:
                    # Try to get content from the response object
                    response_content = str(response)
                
                results = robust_json_parse(response_content)

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

                            # Apply confidence boost and enhanced reasoning
                            original_confidence = result.get("confidence", 0)
                            boosted_confidence = min(95, original_confidence + confidence_boost)
                            
                            # Check if this field had strong pre-reasoning
                            field_data = fields_data[index] if index < len(fields_data) else {}
                            pre_reasoning = field_data.get("pre_reasoning", {})
                            pre_confidence = pre_reasoning.get("confidence", 0)
                            
                            # Use the higher of LLM confidence or pre-reasoning confidence
                            final_confidence = max(boosted_confidence, pre_confidence)
                            
                            # 🚀 AGGRESSIVE: Significantly lower intervention threshold for better UX
                            needs_intervention = result.get("needs_intervention", True)
                            reasoning = result.get("reasoning", "").lower()
                            
                            # Special cases where AI should be more confident
                            is_geographical = any(keyword in reasoning for keyword in ["country", "geographic", "italy", "europe", "schengen", "eea"])
                            has_dummy_data = used_dummy_data or "dummy" in reasoning
                            is_semantic_match = "semantic" in reasoning or "match" in reasoning
                            has_answer = bool(result.get("answer", "").strip())
                            
                            # 🚀 AGGRESSIVE THRESHOLDS - Much lower intervention requirements
                            if (final_confidence >= 60 and has_answer) or \
                               (final_confidence >= 50 and is_geographical) or \
                               (final_confidence >= 50 and has_dummy_data) or \
                               (final_confidence >= 45 and is_semantic_match and has_answer):
                                needs_intervention = False
                                print(f"DEBUG: 🚀 AGGRESSIVE BOOST - Field '{question['field_name']}' confidence: {final_confidence}, geographical: {is_geographical}, dummy: {has_dummy_data}, semantic: {is_semantic_match}, intervention: False")
                            else:
                                print(f"DEBUG: Field '{question['field_name']}' confidence: {final_confidence}, still needs intervention")
                            
                            formatted_result = {
                                "question_id": question["id"],
                                "field_selector": question["field_selector"],
                                "field_name": question["field_name"],
                                "answer": result.get("answer", ""),
                                "confidence": final_confidence,
                                "reasoning": result.get("reasoning", "Batch analysis result") + (f" [Enhanced reasoning confidence: {pre_confidence}]" if pre_confidence > 0 else ""),
                                "needs_intervention": needs_intervention,
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

                # 🚀 METHOD 2 & 3: Apply historical pattern learning and contextual confidence adjustment
                pattern_analysis = self._historical_pattern_learning(questions, profile_data)
                final_results = self._contextual_confidence_adjustment(questions, formatted_results, profile_data)
                
                print(f"DEBUG: Pattern Learning - Found {len(pattern_analysis['successful_geographical_inferences'])} geographic patterns")
                print(f"DEBUG: Pattern Learning - Found {len(pattern_analysis['successful_duration_comparisons'])} duration patterns")

                # 🚀 OPTIMIZATION 3: Save to cache for future use
                self._save_to_cache(cache_key, final_results, "batch")

                return final_results

            except Exception as parse_error:
                print(f"DEBUG: Batch Analysis - Parse error: {str(parse_error)}")
                # Use the response_content we already extracted
                print(f"DEBUG: Raw response: {response_content[:500] if 'response_content' in locals() else str(response)[:500]}...")
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
            if exact_match_ratio >= 0.4:  # 🚀 PERFORMANCE: Lowered to 40% for better early return rate
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

    def _enhanced_contextual_reasoning(self, questions: List[Dict[str, Any]], profile_data: Dict[str, Any],
                                     profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """🚀 NEW: Enhanced contextual reasoning with multi-layer confidence system"""
        
        # Layer 1: Knowledge Base Reasoning
        knowledge_base = {
            "geographical": {
                "eea_countries": [
                    "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic", 
                    "denmark", "estonia", "finland", "france", "germany", "greece", 
                    "hungary", "iceland", "ireland", "italy", "latvia", "liechtenstein", 
                    "lithuania", "luxembourg", "malta", "netherlands", "norway", 
                    "poland", "portugal", "romania", "slovakia", "slovenia", "spain", 
                    "sweden", "switzerland"
                ],
                "commonwealth": ["australia", "canada", "new zealand", "south africa"],
                "english_speaking": ["usa", "united states", "america", "uk", "united kingdom"]
            },
            "business_logic": {
                "visa_duration_ranges": {
                    "short_term": ["1 month", "3 months", "6 months"],
                    "medium_term": ["1 year", "2 years", "3 years"],
                    "long_term": ["5 years", "10 years", "indefinite"]
                }
            }
        }
        
        # Layer 2: Context Enhancement
        enhanced_context = self._build_enhanced_context(profile_data, profile_dummy_data, knowledge_base)
        
        # Layer 3: Reasoning Chain Analysis
        reasoning_chains = []
        for question in questions:
            chain = self._build_reasoning_chain(question, enhanced_context, knowledge_base)
            reasoning_chains.append(chain)
        
        # Calculate confidence boost  
        confidence_boost_result = self._calculate_confidence_boost(reasoning_chains)
        
        return {
            "enhanced_context": enhanced_context,
            "reasoning_chains": reasoning_chains,
            "confidence_boost": confidence_boost_result["confidence_boost"]  # Extract the nested confidence_boost
        }

    def _build_enhanced_context(self, profile_data: Dict[str, Any], profile_dummy_data: Dict[str, Any], 
                              knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Build enhanced context with cross-references and semantic understanding"""
        
        enhanced_context = {
            "direct_data": profile_data,
            "fallback_data": profile_dummy_data or {},
            "inferred_data": {},
            "cross_references": {},
            "semantic_mappings": {}
        }
        
        # Infer additional context from existing data
        if profile_data:
            # Geographic inference
            if "country" in str(profile_data).lower():
                for key, value in self._flatten_dict(profile_data).items():
                    if isinstance(value, str) and any(country in value.lower() for country in knowledge_base["geographical"]["eea_countries"]):
                        enhanced_context["inferred_data"]["is_eea_country"] = True
                        enhanced_context["inferred_data"]["geographic_region"] = "european_economic_area"
                        enhanced_context["cross_references"]["country_to_region"] = value
            
            # Duration inference
            duration_pattern = r'(\d+)\s*(year|month|day)'
            for key, value in self._flatten_dict(profile_data).items():
                if isinstance(value, str):
                    match = re.search(duration_pattern, value.lower())
                    if match:
                        num, unit = match.groups()
                        enhanced_context["inferred_data"][f"{key}_numeric"] = int(num)
                        enhanced_context["inferred_data"][f"{key}_unit"] = unit
        
        return enhanced_context

    def _build_reasoning_chain(self, question: Dict[str, Any], enhanced_context: Dict[str, Any], 
                             knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Build reasoning chain for a specific question"""
        
        field_name = question.get('field_name', '')
        field_type = question.get('field_type', '')
        options = question.get('options', [])
        
        reasoning_chain = {
            "question": question,
            "steps": [],
            "confidence_factors": [],
            "final_confidence": 0,
            "decision_path": []
        }
        
        # Step 1: Direct data matching
        direct_matches = self._find_direct_matches(question, enhanced_context["direct_data"])
        if direct_matches:
            reasoning_chain["steps"].append({
                "type": "direct_match",
                "data": direct_matches,
                "confidence": 95
            })
            reasoning_chain["confidence_factors"].append(95)
        
        # Step 2: Semantic inference
        semantic_matches = self._find_semantic_matches(question, enhanced_context)
        if semantic_matches:
            reasoning_chain["steps"].append({
                "type": "semantic_inference", 
                "data": semantic_matches,
                "confidence": 85
            })
            reasoning_chain["confidence_factors"].append(85)
        
        # Step 3: Knowledge-based reasoning
        knowledge_matches = self._apply_knowledge_reasoning(question, enhanced_context, knowledge_base)
        if knowledge_matches:
            reasoning_chain["steps"].append({
                "type": "knowledge_reasoning",
                "data": knowledge_matches,
                "confidence": 90
            })
            reasoning_chain["confidence_factors"].append(90)
        
        # Calculate final confidence
        if reasoning_chain["confidence_factors"]:
            # Use highest confidence if multiple sources agree
            reasoning_chain["final_confidence"] = max(reasoning_chain["confidence_factors"])
            
            # Boost confidence if multiple sources agree
            if len(reasoning_chain["confidence_factors"]) > 1:
                reasoning_chain["final_confidence"] = min(95, reasoning_chain["final_confidence"] + 10)
        
        return reasoning_chain

    def _find_direct_matches(self, question: Dict[str, Any], data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find direct data matches"""
        field_name = question.get('field_name', '').lower()
        
        # Simple direct matching
        flattened = self._flatten_dict(data)
        for key, value in flattened.items():
            if field_name in key.lower() or key.lower() in field_name:
                return {"key": key, "value": value, "match_type": "direct"}
        
        return None

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _find_semantic_matches(self, question: Dict[str, Any], enhanced_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find semantic matches for a question"""
        field_name = question.get('field_name', '').lower()
        
        # Check inferred data
        inferred_data = enhanced_context.get("inferred_data", {})
        for key, value in inferred_data.items():
            if field_name in key.lower() or any(word in key.lower() for word in field_name.split('_')):
                return {"key": key, "value": value, "match_type": "semantic"}
        
        return None

    def _apply_knowledge_reasoning(self, question: Dict[str, Any], enhanced_context: Dict[str, Any], 
                                 knowledge_base: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply knowledge-based reasoning"""
        field_name = question.get('field_name', '').lower()
        options = question.get('options', [])
        
        # Apply geographical knowledge
        if 'country' in field_name and options:
            geographical = knowledge_base.get('geographical', {})
            for option in options:
                option_text = option.get('text', '').lower()
                if option_text in geographical.get('eea_countries', []):
                    return {"key": "geographical_inference", "value": option_text, "match_type": "knowledge"}
        
        return None

    def _calculate_confidence_boost(self, reasoning_chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence boost based on reasoning chains"""
        total_confidence = 0
        valid_chains = 0
        confident_answers = 0
        
        for chain in reasoning_chains:
            if chain.get('final_confidence', 0) > 0:
                total_confidence += chain['final_confidence']
                valid_chains += 1
                if chain.get('final_confidence', 0) >= 70:
                    confident_answers += 1
        
        avg_confidence = total_confidence / valid_chains if valid_chains > 0 else 0
        overall_boost = min(10, valid_chains * 2)  # Max 10 point boost
        
        return {
            "average_confidence": avg_confidence,
            "valid_chains": valid_chains,
            "total_questions": len(reasoning_chains),
            "confident_answers": confident_answers,
            "confidence_boost": {
                "overall_boost": overall_boost,
                "confident_answers": confident_answers,
                "total_questions": len(reasoning_chains)
            }
        }

    def _find_primary_question(self, grouped_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the primary question from a group of related questions"""
        if not grouped_questions:
            return {}
        
        # Priority order: required fields > text inputs > radio/select > checkboxes
        priority_order = {'text': 4, 'email': 4, 'tel': 4, 'select': 3, 'radio': 2, 'checkbox': 1}
        
        # Sort by priority and required status
        sorted_questions = sorted(grouped_questions, key=lambda q: (
            q.get('required', False),  # Required fields first
            priority_order.get(q.get('field_type', ''), 0),  # Then by field type priority
            -len(q.get('field_label', ''))  # Then by label length (longer labels often more descriptive)
        ), reverse=True)
        
        return sorted_questions[0]

    def _determine_group_answer_type(self, grouped_questions: List[Dict[str, Any]]) -> str:
        """Determine the answer type for a group of questions"""
        if not grouped_questions:
            return "unknown"
        
        field_types = [q.get('field_type', '') for q in grouped_questions]
        
        if 'radio' in field_types:
            return "radio"
        elif 'checkbox' in field_types:
            return "checkbox"  
        elif 'select' in field_types:
            return "select"
        elif any(ft in ['text', 'email', 'tel', 'url', 'number', 'password', 'date', 'time', 'datetime-local'] for ft in field_types):
            return "input"
        else:
            return "unknown"

    def _combine_all_options(self, grouped_questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Combine options from all questions in a group"""
        all_options = []
        seen_values = set()
        
        for question in grouped_questions:
            options = question.get('options', [])
            for option in options:
                value = option.get('value', '')
                if value and value not in seen_values:
                    all_options.append(option)
                    seen_values.add(value)
        
        return all_options

    def _check_answer_consistency(self, merged_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Check consistency of answers across related fields"""
        consistency_issues = []
        
        try:
            # Check for radio button/checkbox group conflicts
            radio_groups = {}
            for item in merged_data:
                question = item.get('question', {})
                field_type = question.get('field_type', '')
                field_name = question.get('field_name', '')
                answer = item.get('answer', '')
                
                # Group radio buttons by name prefix
                if field_type in ['radio', 'checkbox']:
                    group_name = field_name.split('[')[0] if '[' in field_name else field_name
                    if group_name not in radio_groups:
                        radio_groups[group_name] = []
                    radio_groups[group_name].append({
                        'field_name': field_name,
                        'answer': answer,
                        'item': item
                    })
            
            # Check for conflicts within radio groups
            for group_name, group_items in radio_groups.items():
                selected_count = sum(1 for item in group_items if item['answer'] and item['answer'].lower() not in ['false', 'no', ''])
                
                if selected_count > 1:
                    # Multiple selections in radio group
                    consistency_issues.append({
                        'type': 'radio_selection_error',
                        'message': f'Multiple selections in radio group {group_name}',
                        'field_name': group_name,
                        'affected_fields': [item['field_name'] for item in group_items]
                    })
            
            # Check for conflicting values (e.g., "Yes" and "No" to same logical question)
            field_patterns = {}
            for item in merged_data:
                question = item.get('question', {})
                field_name = question.get('field_name', '')
                answer = item.get('answer', '')
                
                # Look for opposing field patterns
                base_name = field_name.lower().replace('_yes', '').replace('_no', '').replace('[yes]', '').replace('[no]', '')
                if base_name not in field_patterns:
                    field_patterns[base_name] = {}
                
                if 'yes' in field_name.lower() or answer.lower() == 'yes':
                    field_patterns[base_name]['yes'] = True
                elif 'no' in field_name.lower() or answer.lower() == 'no':
                    field_patterns[base_name]['no'] = True
            
            # Check for Yes/No conflicts
            for base_name, patterns in field_patterns.items():
                if patterns.get('yes') and patterns.get('no'):
                    consistency_issues.append({
                        'type': 'conflicting_values',
                        'message': f'Conflicting Yes/No answers for {base_name}',
                        'field_name': base_name
                    })
            
            return consistency_issues
            
        except Exception as e:
            print(f"ERROR: Failed to check answer consistency: {str(e)}")
            return [{'type': 'consistency_check_error', 'message': str(e)}]

    def _historical_pattern_learning(self, questions: List[Dict[str, Any]], profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from historical patterns to improve future predictions"""
        try:
            successful_geographical_inferences = []
            successful_duration_comparisons = []
            
            # Simple geographical pattern learning
            for question in questions:
                field_name = question.get('field_name', '').lower()
                if 'country' in field_name or 'region' in field_name:
                    # Check if we have country data that could inform geographical questions
                    for key, value in self._flatten_dict(profile_data).items():
                        if isinstance(value, str) and any(country in value.lower() for country in ["italy", "germany", "france", "spain"]):
                            successful_geographical_inferences.append({
                                'question': question,
                                'pattern': f'Country {value} maps to European region',
                                'confidence': 85
                            })
            
            # Duration comparison pattern learning
            for question in questions:
                field_name = question.get('field_name', '').lower()
                if 'duration' in field_name or 'period' in field_name or 'years' in field_name:
                    options = question.get('options', [])
                    if options:
                        successful_duration_comparisons.append({
                            'question': question,
                            'pattern': 'Duration values need range comparison',
                            'confidence': 80
                        })
            
            return {
                'successful_geographical_inferences': successful_geographical_inferences,
                'successful_duration_comparisons': successful_duration_comparisons,
                'total_patterns': len(successful_geographical_inferences) + len(successful_duration_comparisons)
            }
            
        except Exception as e:
            print(f"ERROR: Failed historical pattern learning: {str(e)}")
            return {
                'successful_geographical_inferences': [],
                'successful_duration_comparisons': [],
                'total_patterns': 0
            }

    def _contextual_confidence_adjustment(self, questions: List[Dict[str, Any]], results: List[Dict[str, Any]], 
                                         profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adjust confidence based on contextual factors"""
        try:
            adjusted_results = []
            
            for result in results:
                adjusted_result = result.copy()
                current_confidence = result.get('confidence', 0)
                
                # Boost confidence for geographic matches
                reasoning = result.get('reasoning', '').lower()
                if any(keyword in reasoning for keyword in ['italy', 'europe', 'geographical', 'country']):
                    adjusted_result['confidence'] = min(95, current_confidence + 10)
                    adjusted_result['reasoning'] += " [Geographic confidence boost]"
                
                # Boost confidence for exact matches
                if 'exact match' in reasoning:
                    adjusted_result['confidence'] = min(98, current_confidence + 5)
                    adjusted_result['reasoning'] += " [Exact match boost]"
                
                # Boost confidence for semantic matches
                if 'semantic' in reasoning:
                    adjusted_result['confidence'] = min(90, current_confidence + 8)
                    adjusted_result['reasoning'] += " [Semantic match boost]"
                
                adjusted_results.append(adjusted_result)
            
            return adjusted_results
            
        except Exception as e:
            print(f"ERROR: Failed contextual confidence adjustment: {str(e)}")
            return results

    def _generate_grouped_question(self, group_fields: List[Dict[str, Any]], base_name: str) -> str:
        """Generate a question for a group of related fields"""
        try:
            if not group_fields:
                return f"Please provide information for {base_name}"
            
            # Use the first field's question as base
            first_field = group_fields[0]
            base_question = first_field.get('question', first_field.get('field_label', base_name))
            
            # If it's already a good question, return it
            if len(base_question) > 10 and '?' in base_question:
                return base_question
            
            # Generate a generic question based on field types
            field_types = [field.get('field_type', '') for field in group_fields]
            
            if 'radio' in field_types or 'checkbox' in field_types:
                return f"Please select the appropriate option for {base_name}?"
            elif 'select' in field_types:
                return f"Please choose from the dropdown for {base_name}?"
            else:
                return f"Please enter the value for {base_name}?"
                
        except Exception as e:
            print(f"ERROR: Failed to generate grouped question: {str(e)}")
            return f"Please provide information for {base_name}"

    def _validate_and_improve_action_selector(self, action: Dict[str, Any], html_content: str, 
                                             current_attempt: int = 1) -> Dict[str, Any]:
        """Validate and improve action selector"""
        try:
            selector = action.get('selector', '')
            action_type = action.get('action_type', '')
            
            # Basic validation - if selector exists in HTML
            if selector and selector in html_content:
                return action
            
            # Try to improve the selector
            field_name = action.get('field_name', '')
            if field_name:
                # Try different selector patterns
                improved_selectors = [
                    f'input[name="{field_name}"]',
                    f'select[name="{field_name}"]',
                    f'textarea[name="{field_name}"]',
                    f'[name="{field_name}"]',
                    f'#{field_name}',
                    f'.{field_name}'
                ]
                
                for improved_selector in improved_selectors:
                    if improved_selector in html_content:
                        action['selector'] = improved_selector
                        action['validation_improved'] = True
                        return action
            
            # If we can't improve it, mark as needs manual review
            action['needs_manual_review'] = True
            action['validation_failed'] = True
            return action
            
        except Exception as e:
            print(f"ERROR: Failed to validate action selector: {str(e)}")
            action['validation_error'] = str(e)
            return action

    def _map_field_type_to_answer_type(self, field_type: str) -> str:
        """Map field type to answer type"""
        mapping = {
            'text': 'text',
            'email': 'email',
            'tel': 'phone',
            'number': 'number',
            'date': 'date',
            'select': 'option',
            'radio': 'option',
            'checkbox': 'boolean',
            'textarea': 'text',
            'file': 'file',
            'hidden': 'hidden'
        }
        return mapping.get(field_type.lower(), 'text')

    async def _generate_ai_answer_async(self, question: Dict[str, Any], profile_data: Dict[str, Any], 
                                       profile_dummy_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate AI answer for a single question (async version)"""
        try:
            # For now, fall back to synchronous batch processing with single question
            results = await self._batch_analyze_fields_async([question], profile_data, profile_dummy_data)
            
            if results and len(results) > 0:
                return results[0]
            else:
                # Return empty result
                return {
                    "question_id": question.get("id", ""),
                    "field_selector": question.get("field_selector", ""),
                    "field_name": question.get("field_name", ""),
                    "answer": "",
                    "confidence": 0,
                    "reasoning": "No AI answer generated",
                    "needs_intervention": True,
                    "data_source_path": "",
                    "field_match_type": "none",
                    "used_dummy_data": False,
                    "source_name": ""
                }
                
        except Exception as e:
            print(f"ERROR: Failed to generate AI answer async: {str(e)}")
            return {
                "question_id": question.get("id", ""),
                "field_selector": question.get("field_selector", ""),
                "field_name": question.get("field_name", ""),
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error generating answer: {str(e)}",
                "needs_intervention": True,
                "data_source_path": "",
                "field_match_type": "error",
                "used_dummy_data": False,
                "source_name": ""
            }
