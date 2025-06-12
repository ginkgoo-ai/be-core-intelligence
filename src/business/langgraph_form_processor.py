from typing import Dict, List, Any, Optional, Literal, TypedDict
from datetime import datetime
import json
import uuid
import os
from bs4 import BeautifulSoup

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from sqlalchemy.orm import Session

from src.database.workflow_repositories import (
    WorkflowInstanceRepository, StepInstanceRepository
)
from src.model.workflow_entities import StepStatus

class FormAnalysisState(TypedDict):
    """State for form analysis workflow"""
    workflow_id: str
    step_key: str
    form_html: str
    profile_data: Dict[str, Any]
    parsed_form: Optional[Dict[str, Any]]
    detected_fields: List[Dict[str, Any]]
    field_questions: List[Dict[str, Any]]
    ai_answers: List[Dict[str, Any]]
    merged_qa_data: List[Dict[str, Any]]  # New field for merged question-answer data
    form_actions: List[Dict[str, Any]]
    analysis_complete: bool
    error_message: Optional[str]
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

class LangGraphFormProcessor:
    """LangGraph-based form processor for workflow integration"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.instance_repo = WorkflowInstanceRepository(db_session)
        self.step_repo = StepInstanceRepository(db_session)
        
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
        workflow.add_edge("action_generator", "result_saver")
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
    
    def process_form(self, workflow_id: str, step_key: str, form_html: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process form using LangGraph workflow"""
        try:
            # Initialize state
            initial_state = FormAnalysisState(
                workflow_id=workflow_id,
                step_key=step_key,
                form_html=form_html,
                profile_data=profile_data,
                parsed_form=None,
                detected_fields=[],
                field_questions=[],
                ai_answers=[],
                merged_qa_data=[],
                form_actions=[],
                analysis_complete=False,
                error_message=None,
                messages=[]
            )
            
            # Run workflow
            config = {"configurable": {"thread_id": f"{workflow_id}_{step_key}_{uuid.uuid4()}"}}
            result = self.app.invoke(initial_state, config)
            
            if result.get("analysis_complete", False):
                # Format actions according to Google plugin requirements
                formatted_actions = []
                for action in result.get("form_actions", []):
                    formatted_action = {
                        "selector": action["selector"],
                        "type": action["action_type"]  # 改为type字段
                    }
                    # 只在有值的时候添加value字段
                    if action.get("value"):
                        formatted_action["value"] = action["value"]
                    
                    formatted_actions.append(formatted_action)
                
                # Format merged Q&A data according to your requirements
                # [{"question":{"name":"where are you come from"},"answer":[{"value":"china", "check":1},{"value":"japan"}]}]
                formatted_qa_data = []
                for qa in result.get("merged_qa_data", []):
                    qa_item = {
                        "question": {
                            "name": qa.get("question", "")  # Only include the question text
                        },
                        "answer": []
                    }
                    
                    # Handle different field types
                    if qa.get("field_type") in ["select", "radio", "checkbox"] and qa.get("options"):
                        # For fields with options, create answer list with check status
                        ai_answer = qa.get("answer", "")
                        for option in qa.get("options", []):
                            answer_item = {
                                "value": option.get("value", ""),
                                "text": option.get("text", option.get("value", ""))
                            }
                            # Mark as checked if it matches AI answer
                            if ai_answer and (str(ai_answer).lower() == str(option.get("value", "")).lower() or 
                                            str(ai_answer).lower() == str(option.get("text", "")).lower()):
                                answer_item["check"] = 1
                            
                            qa_item["answer"].append(answer_item)
                    else:
                        # For text inputs, just use the AI answer directly
                        if qa.get("answer"):
                            qa_item["answer"].append({
                                "value": qa.get("answer", ""),
                                "check": 1
                            })
                    
                    formatted_qa_data.append(qa_item)
                
                # 返回简化的格式
                return {
                    "success": True,
                    "actions": formatted_actions,  # Google插件格式
                    "data": formatted_qa_data  # 您要求的新格式
                }
            else:
                return {
                    "success": False,
                    "actions": [],
                    "data": [],
                    "error": result.get("error_message", "Analysis failed")
                }
            
        except Exception as e:
            return {
                "success": False,
                "actions": [],
                "data": [],
                "error": str(e)
            }
    
    def _html_parser_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 1: Parse HTML and extract form structure"""
        try:
            soup = BeautifulSoup(state["form_html"], 'html.parser')
            
            # Find forms
            forms = soup.find_all('form')
            if not forms:
                # Treat entire HTML as form content
                parsed_form = {
                    'action': '',
                    'method': 'post',
                    'elements': soup
                }
            else:
                form = forms[0]
                parsed_form = {
                    'action': form.get('action', ''),
                    'method': form.get('method', 'post'),
                    'elements': form
                }
            
            state["parsed_form"] = parsed_form
            state["messages"].append({
                "type": "system",
                "content": f"HTML parsed successfully. Found {len(forms)} form(s)."
            })
            
        except Exception as e:
            state["error_message"] = f"HTML parsing failed: {str(e)}"
        
        return state
    
    def _field_detector_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 2: Detect and analyze form fields"""
        try:
            if not state["parsed_form"]:
                raise ValueError("No parsed form available")
            
            form_elements = state["parsed_form"]["elements"]
            detected_fields = []
            
            # Find all input elements including radio and checkbox groups
            inputs = form_elements.find_all(['input', 'select', 'textarea'])
            
            # Track radio groups to avoid duplicates
            radio_groups = {}
            
            for element in inputs:
                field_info = self._extract_field_info(element)
                if field_info:
                    # Special handling for radio buttons - group them by name
                    if field_info.get("type") == "radio":
                        name = field_info.get("name")
                        if name:
                            if name not in radio_groups:
                                radio_groups[name] = field_info
                                # Find all radio options for this group
                                radio_options = []
                                all_radios = form_elements.find_all('input', {'type': 'radio', 'name': name})
                                for radio in all_radios:
                                    label_text = self._find_field_label(radio)
                                    radio_options.append({
                                        "value": radio.get("value", ""),
                                        "text": label_text or radio.get("value", "")
                                    })
                                radio_groups[name]["options"] = radio_options
                                detected_fields.append(radio_groups[name])
                    else:
                        detected_fields.append(field_info)
            
            state["detected_fields"] = detected_fields
            state["messages"].append({
                "type": "system", 
                "content": f"Detected {len(detected_fields)} form fields."
            })
            
        except Exception as e:
            state["error_message"] = f"Field detection failed: {str(e)}"
        
        return state
    
    def _question_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 3: Generate questions for each form field"""
        try:
            questions = []
            
            for field in state["detected_fields"]:
                question = self._generate_field_question(field)
                if question:
                    questions.append(question)
            
            state["field_questions"] = questions
            state["messages"].append({
                "type": "system",
                "content": f"Generated {len(questions)} field questions."
            })
            
        except Exception as e:
            state["error_message"] = f"Question generation failed: {str(e)}"
        
        return state
    
    def _profile_retriever_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 4: Profile data is already available in state (no DB retrieval needed)"""
        try:
            # Profile data is already passed in via parameters, no DB query needed
            state["messages"].append({
                "type": "system",
                "content": f"Profile data available with {len(state['profile_data'])} sections."
            })
            
        except Exception as e:
            state["error_message"] = f"Profile processing failed: {str(e)}"
        
        return state
    
    def _ai_answerer_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 5: Use AI to answer field questions based on profile"""
        try:
            answers = []
            
            for question in state["field_questions"]:
                answer = self._generate_ai_answer(question, state["profile_data"])
                if answer:
                    answers.append(answer)
            
            state["ai_answers"] = answers
            state["messages"].append({
                "type": "system",
                "content": f"Generated {len(answers)} AI answers."
            })
            
        except Exception as e:
            state["error_message"] = f"AI answering failed: {str(e)}"
        
        return state
    
    def _qa_merger_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 6: Merge question and answer data"""
        try:
            merged_qa_data = []
            
            for question in state["field_questions"]:
                answer = self._find_answer_for_question(question, state["ai_answers"])
                # Always create merged data, even if no answer is found
                merged_item = {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "field_type": question["field_type"],
                    "field_label": question["field_label"],
                    "question": question["question"],
                    "required": question["required"],
                    "options": question["options"],
                    "answer": answer["answer"] if answer else "",
                    "confidence": answer["confidence"] if answer else 0,
                    "reasoning": answer["reasoning"] if answer else "No answer found"
                }
                merged_qa_data.append(merged_item)
            
            state["merged_qa_data"] = merged_qa_data
            state["messages"].append({
                "type": "system",
                "content": f"Merged {len(merged_qa_data)} question-answer pairs."
            })
            
        except Exception as e:
            state["error_message"] = f"QA merging failed: {str(e)}"
        
        return state
    
    def _action_generator_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 7: Generate form actions based on merged question-answer data"""
        try:
            actions = []
            
            for data in state["merged_qa_data"]:
                action = self._generate_form_action(data)
                if action:
                    actions.append(action)
            
            # Sort actions by order
            actions.sort(key=lambda x: x.get("order", 0))
            
            state["form_actions"] = actions
            state["messages"].append({
                "type": "system",
                "content": f"Generated {len(actions)} form actions."
            })
            
        except Exception as e:
            state["error_message"] = f"Action generation failed: {str(e)}"
        
        return state
    
    def _result_saver_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 8: Save results to database"""
        try:
            # Get step instance
            step = self.step_repo.get_step_by_key(state["workflow_id"], state["step_key"])
            if not step:
                raise ValueError(f"Step {state['step_key']} not found")
            
            # Helper function to make data JSON serializable
            def make_serializable(obj):
                """Convert objects to JSON serializable format"""
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif hasattr(obj, 'get_text'):  # BeautifulSoup object
                    return str(obj)
                elif hasattr(obj, 'prettify'):  # BeautifulSoup object
                    return obj.prettify()
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif obj is None:
                    return None
                else:
                    try:
                        # Try to convert to string if it's not a basic type
                        if not isinstance(obj, (str, int, float, bool)):
                            return str(obj)
                        return obj
                    except:
                        return str(obj)
            
            # Prepare simplified data for saving - only store what's necessary
            simplified_data = []
            for qa in state["merged_qa_data"]:
                qa_item = {
                    "question": {
                        "name": qa.get("question", "")  # Only include the question text
                    },
                    "answer": []
                }
                
                # Handle different field types
                if qa.get("field_type") in ["select", "radio", "checkbox"] and qa.get("options"):
                    # For fields with options, create answer list with check status
                    ai_answer = qa.get("answer", "")
                    for option in qa.get("options", []):
                        answer_item = {
                            "value": option.get("value", ""),
                            "text": option.get("text", option.get("value", ""))
                        }
                        # Mark as checked if it matches AI answer
                        if ai_answer and (str(ai_answer).lower() == str(option.get("value", "")).lower() or 
                                        str(ai_answer).lower() == str(option.get("text", "")).lower()):
                            answer_item["check"] = 1
                        
                        qa_item["answer"].append(answer_item)
                else:
                    # For text inputs, just use the AI answer directly
                    if qa.get("answer"):
                        qa_item["answer"].append({
                            "value": qa.get("answer", ""),
                            "check": 1
                        })
                
                simplified_data.append(qa_item)
            
            # Prepare data to save - simplified structure
            analysis_data = {
                "data": simplified_data,  # This is the main data in your requested format
                "actions": make_serializable(state["form_actions"]),
                "metadata": {
                    "processed_at": datetime.utcnow().isoformat(),
                    "workflow_id": state["workflow_id"],
                    "step_key": state["step_key"],
                    "success": True,
                    "qa_count": len(simplified_data),
                    "action_count": len(state["form_actions"])
                }
            }
            
            # Update step data
            self.step_repo.update_step_data(step.step_instance_id, analysis_data)
            
            self.db.commit()
            
            state["analysis_complete"] = True
            state["messages"].append({
                "type": "system",
                "content": "Analysis results saved to database successfully."
            })
            
        except Exception as e:
            self.db.rollback()
            state["error_message"] = f"Result saving failed: {str(e)}"
        
        return state
    
    def _error_handler_node(self, state: FormAnalysisState) -> FormAnalysisState:
        """Node 9: Handle errors and save error information"""
        try:
            # Get step instance
            step = self.step_repo.get_step_by_key(state["workflow_id"], state["step_key"])
            if step:
                # Helper function to make data JSON serializable
                def make_serializable(obj):
                    """Convert objects to JSON serializable format"""
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    elif hasattr(obj, 'get_text'):  # BeautifulSoup object
                        return str(obj)
                    elif hasattr(obj, 'prettify'):  # BeautifulSoup object
                        return obj.prettify()
                    elif isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(item) for item in obj]
                    elif obj is None:
                        return None
                    else:
                        try:
                            # Try to convert to string if it's not a basic type
                            if not isinstance(obj, (str, int, float, bool)):
                                return str(obj)
                            return obj
                        except:
                            return str(obj)
                
                # Prepare simplified error data
                simplified_data = []
                if state.get("merged_qa_data"):
                    for qa in state.get("merged_qa_data"):
                        qa_item = {
                            "question": {
                                "name": qa.get("question", "")  # Only include the question text
                            },
                            "answer": []
                        }
                        
                        # Handle different field types
                        if qa.get("field_type") in ["select", "radio", "checkbox"] and qa.get("options"):
                            ai_answer = qa.get("answer", "")
                            for option in qa.get("options", []):
                                answer_item = {
                                    "value": option.get("value", ""),
                                    "text": option.get("text", option.get("value", ""))
                                }
                                if ai_answer and (str(ai_answer).lower() == str(option.get("value", "")).lower() or 
                                                str(ai_answer).lower() == str(option.get("text", "")).lower()):
                                    answer_item["check"] = 1
                                qa_item["answer"].append(answer_item)
                        else:
                            if qa.get("answer"):
                                qa_item["answer"].append({
                                    "value": qa.get("answer", ""),
                                    "check": 1
                                })
                        
                        simplified_data.append(qa_item)
                
                error_data = {
                    "error": {
                        "message": state["error_message"],
                        "occurred_at": datetime.utcnow().isoformat(),
                        "workflow_id": state["workflow_id"],
                        "step_key": state["step_key"]
                    },
                    "data": simplified_data,  # Use same format as success case
                    "actions": make_serializable(state.get("form_actions", []))
                }
                
                # Update step data first
                self.step_repo.update_step_data(step.step_instance_id, error_data)
                
                # Set error details and status
                self.step_repo.set_step_error(step.step_instance_id, state["error_message"])
                
                self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            print(f"Error handler failed: {e}")
        
        return state
    
    def _check_for_errors(self, state: FormAnalysisState) -> Literal["continue", "error"]:
        """Check if there are errors in the state"""
        return "error" if state.get("error_message") else "continue"
    
    def _extract_field_info(self, element) -> Optional[Dict[str, Any]]:
        """Extract information from HTML form element"""
        try:
            field_type = element.name.lower()
            
            # Handle input types more specifically
            if element.name == "input":
                field_type = element.get("type", "text").lower()
            
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
            
            # Only return fields that have a name or meaningful selector
            return field_info if (field_info["name"] or field_info["label"]) else None
            
        except Exception:
            return None
    
    def _generate_selector(self, element) -> str:
        """Generate CSS selector for element"""
        # Start with element type
        element_type = element.name
        
        # Build selector based on available attributes
        if element.get("id"):
            return f"#{element.get('id')}"
        
        if element.get("name"):
            if element.name == "input" and element.get("type"):
                return f"input[type='{element.get('type')}'][name='{element.get('name')}']"
            else:
                return f"{element_type}[name='{element.get('name')}']"
        
        # Fallback to class-based selector
        if element.get("class"):
            class_names = element.get("class")
            if isinstance(class_names, list):
                class_str = ".".join(class_names)
            else:
                class_str = class_names
            return f"{element_type}.{class_str}"
        
        # Last resort - use element type only (not very reliable)
        return element_type
    
    def _find_field_label(self, element) -> str:
        """Find label text for form field"""
        # Try to find associated label
        field_id = element.get("id")
        if field_id:
            label = element.find_previous("label", {"for": field_id})
            if label:
                return label.get_text(strip=True)
        
        # Try to find nearby label
        label = element.find_previous("label")
        if label:
            return label.get_text(strip=True)
        
        return element.get("placeholder", "")
    
    def _generate_field_question(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a question for a form field"""
        field_name = field.get("name", "")
        field_label = field.get("label", "")
        field_type = field.get("type", "")
        placeholder = field.get("placeholder", "")
        
        # Extract just the essential question text from HTML
        question_text = ""
        if field_label:
            question_text = field_label.strip()
        elif placeholder:
            question_text = placeholder.strip()
        elif field_name:
            question_text = field_name.replace("_", " ").replace("-", " ").title()
        else:
            question_text = f"{field_type} field"
        
        return {
            "id": f"q_{field_name}_{uuid.uuid4().hex[:8]}",
            "field_selector": field["selector"],
            "field_name": field_name,
            "field_type": field_type,
            "field_label": field_label,
            "question": question_text,  # Simplified to just the text
            "required": field.get("required", False),
            "options": field.get("options", [])
        }
    
    def _generate_ai_answer(self, question: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI answer for a field question"""
        try:
            # Create prompt for AI
            prompt = f"""
            # Role 
            You are a backend of a Google plugin, analyzing the html web pages sent from the front end, 
            analyzing the form items that need to be filled in them, retrieving the customer data that has been provided, 
            filling in the form content, and returning it to the front end in a fixed json format.

            # Task
            Based on the user profile data, determine the appropriate value for this form field:
            
            Field Name: {question['field_name']}
            Field Type: {question['field_type']}
            Field Label: {question['field_label']}
            Field Selector: {question['field_selector']}
            Required: {question['required']}
            Question: {question['question']}
            
            # User Profile Data:
            {json.dumps(profile, indent=2)}
            
            # Instructions
            1. Find the most appropriate value from the user profile data for this field
            2. Consider the field type and requirements
            3. If no suitable data is found, leave answer empty but provide reasoning
            4. Provide confidence level (0-100) based on data quality match
            
            # Response Format (JSON only):
            {{
                "answer": "your answer here",
                "confidence": 85,
                "reasoning": "explanation of why this answer was chosen"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                # Try to parse JSON response
                result = json.loads(response.content)
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "answer": result.get("answer", ""),
                    "confidence": result.get("confidence", 0),
                    "reasoning": result.get("reasoning", "")
                }
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "question_id": question["id"],
                    "field_selector": question["field_selector"],
                    "field_name": question["field_name"],
                    "answer": response.content,
                    "confidence": 50,
                    "reasoning": "AI response could not be parsed as JSON"
                }
                
        except Exception as e:
            return {
                "question_id": question["id"],
                "field_selector": question["field_selector"],
                "field_name": question["field_name"],
                "answer": "",
                "confidence": 0,
                "reasoning": f"Error generating answer: {str(e)}"
            }
    
    def _generate_form_action(self, merged_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate form action from merged question-answer data"""
        # Check if we have a valid answer with reasonable confidence
        answer = merged_data.get("answer", "")
        confidence = merged_data.get("confidence", 0)
        
        if not answer or confidence < 30:
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
            # For text inputs, textarea, etc.
            action_type = "input"
            return {
                "selector": selector,
                "action_type": action_type,
                "value": answer,
                "field_name": field_name,
                "confidence": confidence,
                "reasoning": merged_data.get("reasoning", ""),
                "order": hash(field_name) % 1000
            }
        
        # If we get here, no action could be generated
        return None

    def _find_answer_for_question(self, question: Dict[str, Any], answers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the answer for a given question"""
        for answer in answers:
            if answer["field_selector"] == question["field_selector"]:
                return answer
        return None 