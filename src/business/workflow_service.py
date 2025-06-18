from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import uuid

from src.database.workflow_repositories import (
    WorkflowDefinitionRepository, WorkflowInstanceRepository,
    StepInstanceRepository
)
from src.model.workflow_entities import WorkflowStatus, StepStatus
from src.model.workflow_schemas import (
    WorkflowInitiationPayload, WorkflowInstanceSummary, WorkflowInstanceDetail,
    StepDataModel, PersonalDetailsModel, ContactAddressModel, StepStatusUpdate,
    NextStepInfo, WorkflowStatus as WorkflowStatusSchema, AutosaveConfirmation,
    FormProcessResult, FormActionModel, StepInstanceDetail, ProgressFileUploadResult
)
from src.business.langgraph_form_processor import LangGraphFormProcessor
from src.model.profile_schema import Profile

class WorkflowService:
    """Main workflow service implementing visa application process"""
    
    # New 15 steps of visa application workflow
    DEFAULT_VISA_WORKFLOW_STEPS = [
        {"key": "linking_code", "name": "Linking code for family members", "order": 1},
        {"key": "immigration_adviser", "name": "Immigration adviser details", "order": 2},
        {"key": "contact_preferences", "name": "Contact preferences", "order": 3},
        {"key": "other_names", "name": "Other names and nationalities", "order": 4},
        {"key": "people_applying", "name": "People applying with you", "order": 5},
        {"key": "your_location", "name": "Your location", "order": 6},
        {"key": "work_details", "name": "Work details", "order": 7},
        {"key": "personal_details", "name": "Personal details", "order": 8},
        {"key": "family_relationships", "name": "Family and relationships", "order": 9},
        {"key": "travel_history", "name": "Travel history", "order": 10},
        {"key": "criminality", "name": "Criminality", "order": 11},
        {"key": "financial_maintenance", "name": "Financial maintenance", "order": 12},
        {"key": "english_language", "name": "English language ability", "order": 13},
        {"key": "security_questions", "name": "Account security questions", "order": 14},
        {"key": "declaration", "name": "Declaration", "order": 15}
    ]
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.definition_repo = WorkflowDefinitionRepository(db_session)
        self.instance_repo = WorkflowInstanceRepository(db_session)
        self.step_repo = StepInstanceRepository(db_session)
        self.form_processor = LangGraphFormProcessor(db_session)
    
    def create_workflow(self, payload: WorkflowInitiationPayload) -> WorkflowInstanceSummary:
        """Create a new workflow instance"""
        try:
            # Get step definitions from WorkflowDefinition or use default
            step_definitions = self.DEFAULT_VISA_WORKFLOW_STEPS  # Default fallback
            
            if payload.workflow_definition_id:
                # Get workflow definition from database
                workflow_def = self.definition_repo.get_definition_by_id(payload.workflow_definition_id)
                if workflow_def and workflow_def.step_definitions:
                    step_definitions = workflow_def.step_definitions
                    print(f"✅ 使用工作流定义ID {payload.workflow_definition_id} 的步骤配置")
                else:
                    print(f"⚠️ 工作流定义未找到或无步骤配置，使用默认15步配置")
            else:
                # Create a default workflow definition if none specified
                workflow_def = self.definition_repo.create_definition(
                    name="Default Visa Application Workflow",
                    description="Default 15-step visa application process",
                    step_definitions=step_definitions
                )
                payload.workflow_definition_id = workflow_def.workflow_definition_id
                print(f"✅ 创建默认工作流定义: {workflow_def.workflow_definition_id}")
            
            # Create workflow instance
            instance = self.instance_repo.create_instance(
                user_id=payload.user_id,
                workflow_definition_id=payload.workflow_definition_id
            )
            
            # Create step instances from definition
            steps = self.step_repo.create_steps_from_definition(
                workflow_instance_id=instance.workflow_instance_id,
                step_definitions=step_definitions
            )
            
            print(f"✅ 创建了 {len(steps)} 个步骤")
            
            # Set first step as active
            if steps:
                first_step = min(steps, key=lambda s: s.order or 0)
                self.step_repo.update_step_status(
                    first_step.step_instance_id,
                    StepStatus.ACTIVE
                )
                self.instance_repo.update_instance_status(
                    instance.workflow_instance_id,
                    WorkflowStatus.IN_PROGRESS,
                    first_step.step_key
                )
                print(f"✅ 激活第一步: {first_step.name} ({first_step.step_key})")
            
            self.db.commit()
            
            return WorkflowInstanceSummary(
                workflow_instance_id=instance.workflow_instance_id,
                user_id=instance.user_id,
                status=instance.status,
                current_step_key=instance.current_step_key,
                created_at=instance.created_at,
                updated_at=instance.updated_at,
                completed_at=instance.completed_at
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_workflow_status(self, workflow_id: str) -> WorkflowInstanceDetail:
        """Get detailed workflow status"""
        instance = self.instance_repo.get_instance_by_id(workflow_id)
        if not instance:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Get all steps
        steps = self.step_repo.get_workflow_steps(workflow_id)
        
        # Build step details (simple flat structure)
        step_details = []
        for step in sorted(steps, key=lambda s: s.order or 0):
            step_detail = StepInstanceDetail(
                step_instance_id=step.step_instance_id,
                workflow_instance_id=step.workflow_instance_id,
                step_key=step.step_key,
                name=step.name,
                order=step.order,
                status=step.status,
                data=step.data,
                next_step_url=step.next_step_url,
                started_at=step.started_at,
                completed_at=step.completed_at,
                error_details=step.error_details
            )
            step_details.append(step_detail)
        
        return WorkflowInstanceDetail(
            workflow_instance_id=instance.workflow_instance_id,
            user_id=instance.user_id,
            status=instance.status,
            current_step_key=instance.current_step_key,
            created_at=instance.created_at,
            updated_at=instance.updated_at,
            completed_at=instance.completed_at,
            workflow_definition_id=instance.workflow_definition_id,
            progress_file_id=instance.progress_file_id,
            dummy_data_usage=instance.dummy_data_usage,
            steps=step_details
        )
    
    def pause_workflow(self, workflow_id: str) -> WorkflowStatusSchema:
        """Pause workflow execution"""
        try:
            instance = self.instance_repo.pause_instance(workflow_id)
            if not instance:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            self.db.commit()
            
            return WorkflowStatusSchema(
                workflow_instance_id=workflow_id,
                status=instance.status,
                message="工作流已暂停",
                updated_at=instance.updated_at
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def resume_workflow(self, workflow_id: str) -> WorkflowStatusSchema:
        """Resume workflow execution"""
        try:
            instance = self.instance_repo.resume_instance(workflow_id)
            if not instance:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            self.db.commit()
            
            return WorkflowStatusSchema(
                workflow_instance_id=workflow_id,
                status=instance.status,
                message="工作流已恢复",
                updated_at=instance.updated_at
            )
            
        except Exception as e:
            self.db.rollback()
            raise e

    def _get_next_step(self, workflow_id: str, current_step_key: str) -> Dict[str, Any]:
        """Get next step info based on current step"""
        steps = self.step_repo.get_workflow_steps(workflow_id)
        sorted_steps = sorted(steps, key=lambda s: s.order or 0)
        
        # Find current step index
        current_index = -1
        for i, step in enumerate(sorted_steps):
            if step.step_key == current_step_key:
                current_index = i
                break
        
        # If we have a next step
        if current_index >= 0 and current_index < len(sorted_steps) - 1:
            next_step = sorted_steps[current_index + 1]
            return {
                "step_key": next_step.step_key,
                "step_name": next_step.name,
                "next_step_url": f"/workflow/{workflow_id}/step/{next_step.step_key}"
            }
        
        # No more steps - workflow should complete
        return {"workflow_complete": True}
    
    def upload_progress_file(self, workflow_id: str, file_id: str) -> dict:
        """Upload progress file for workflow instance"""
        try:
            # Check if workflow instance exists
            instance = self.instance_repo.get_instance_by_id(workflow_id)
            if not instance:
                raise ValueError(f"工作流实例 {workflow_id} 不存在")
            
            # Update the progress file ID
            updated_instance = self.instance_repo.update_progress_file_id(workflow_id, file_id)
            if not updated_instance:
                raise ValueError(f"更新工作流实例进度文件失败")
            
            self.db.commit()
            
            return ProgressFileUploadResult(
                success=True,
                message="进度文件上传成功",
                workflow_instance_id=workflow_id,
                progress_file_id=file_id
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def get_progress_file(self, workflow_id: str) -> dict:
        """Get progress file information for workflow instance"""
        try:
            instance = self.instance_repo.get_instance_by_id(workflow_id)
            if not instance:
                raise ValueError(f"工作流实例 {workflow_id} 不存在")
            
            return {
                "workflow_instance_id": workflow_id,
                "progress_file_id": instance.progress_file_id,
                "has_progress_file": instance.progress_file_id is not None,
                "updated_at": instance.updated_at.isoformat() if instance.updated_at else None
            }
            
        except Exception as e:
            raise e
    
    def get_user_workflows(self, user_id: str, limit: int = 50) -> List[WorkflowInstanceSummary]:
        """Get user's workflow instances list"""
        try:
            instances = self.instance_repo.get_user_instances(user_id, limit)
            
            # Convert to WorkflowInstanceSummary list
            workflow_summaries = []
            for instance in instances:
                summary = WorkflowInstanceSummary(
                    workflow_instance_id=instance.workflow_instance_id,
                    user_id=instance.user_id,
                    status=instance.status,
                    current_step_key=instance.current_step_key,
                    created_at=instance.created_at,
                    updated_at=instance.updated_at,
                    completed_at=instance.completed_at
                )
                workflow_summaries.append(summary)
            
            return workflow_summaries
            
        except Exception as e:
            raise e

class StepService:
    """Step management service"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.instance_repo = WorkflowInstanceRepository(db_session)
        self.step_repo = StepInstanceRepository(db_session)
        self.form_processor = LangGraphFormProcessor(db_session)
    
    def get_step_data(self, workflow_id: str, step_key: str) -> StepDataModel:
        """Get step data"""
        step = self.step_repo.get_step_by_key(workflow_id, step_key)
        if not step:
            raise ValueError(f"Step {step_key} not found in workflow {workflow_id}")
        
        # Ensure step.data is not None, use empty dict as fallback
        step_data = step.data or {}
        
        # For personal_details and contact_address, extract user-friendly data from complex JSON structure
        return StepDataModel(
            step_key=step_key,
            data=step_data
        )

    
    def _extract_personal_details_from_json(self, step_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract personal details fields from complex JSON structure"""
        extracted_data = {}
        
        # Check if data contains form_data (new LangGraph format)
        form_data = step_data.get("form_data", [])
        
        # Mapping of field names to PersonalDetailsModel attributes
        field_mapping = {
            "given_name": ["given_name", "first_name", "firstname", "givenname"],
            "family_name": ["family_name", "last_name", "lastname", "surname", "familyname"],
            "date_of_birth": ["date_of_birth", "dob", "birth_date", "dateofbirth"],
            "nationality": ["nationality", "country", "citizenship"],
            "passport_number": ["passport_number", "passport", "passportno", "passport_no"]
        }
        
        # Extract from form_data if available
        if form_data:
            for item in form_data:
                question_data = item.get("question", {}).get("data", {})
                field_name = question_data.get("name", "").lower()
                
                # Get the answer value
                answer_data = item.get("question", {}).get("answer", {}).get("data", [])
                if answer_data and len(answer_data) > 0:
                    value = answer_data[0].get("value", "")
                    
                    # Map field name to PersonalDetailsModel attribute
                    for attr_name, possible_names in field_mapping.items():
                        if any(name in field_name for name in possible_names):
                            extracted_data[attr_name] = value
                            break
        
        # Fallback: check if data is directly stored in step_data (legacy format)
        for attr_name, possible_names in field_mapping.items():
            if attr_name not in extracted_data:
                for possible_name in possible_names:
                    if possible_name in step_data:
                        extracted_data[attr_name] = step_data[possible_name]
                        break
        
        return extracted_data
    
    def _extract_contact_address_from_json(self, step_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract contact address fields from complex JSON structure"""
        extracted_data = {}
        
        # Check if data contains form_data (new LangGraph format)
        form_data = step_data.get("form_data", [])
        
        # Mapping of field names to ContactAddressModel attributes
        field_mapping = {
            "email": ["email", "email_address", "e_mail"],
            "phone": ["phone", "telephone", "mobile", "phone_number"],
            "address_line1": ["address_line1", "address1", "street", "address"],
            "address_line2": ["address_line2", "address2", "apartment", "unit"],
            "city": ["city", "town", "locality"],
            "postal_code": ["postal_code", "postcode", "zip", "zipcode"]
        }
        
        # Extract from form_data if available
        if form_data:
            for item in form_data:
                question_data = item.get("question", {}).get("data", {})
                field_name = question_data.get("name", "").lower()
                
                # Get the answer value
                answer_data = item.get("question", {}).get("answer", {}).get("data", [])
                if answer_data and len(answer_data) > 0:
                    value = answer_data[0].get("value", "")
                    
                    # Map field name to ContactAddressModel attribute
                    for attr_name, possible_names in field_mapping.items():
                        if any(name in field_name for name in possible_names):
                            extracted_data[attr_name] = value
                            break
        
        # Fallback: check if data is directly stored in step_data (legacy format)
        for attr_name, possible_names in field_mapping.items():
            if attr_name not in extracted_data:
                for possible_name in possible_names:
                    if possible_name in step_data:
                        extracted_data[attr_name] = step_data[possible_name]
                        break
        
        return extracted_data
    
    def get_step_raw_data(self, workflow_id: str, step_key: str) -> Dict[str, Any]:
        """Get step raw JSON data from database without any transformation"""
        step = self.step_repo.get_step_by_key(workflow_id, step_key)
        if not step:
            raise ValueError(f"Step {step_key} not found in workflow {workflow_id}")
        
        return {
            "step_instance_id": step.step_instance_id,
            "workflow_instance_id": step.workflow_instance_id,
            "step_key": step.step_key,
            "name": step.name,
            "status": step.status.value,
            "order": step.order,
            "data": step.data or {},
            "started_at": step.started_at.isoformat() if step.started_at else None,
            "completed_at": step.completed_at.isoformat() if step.completed_at else None,
            "updated_at": step.updated_at.isoformat() if step.updated_at else None,
            "error_details": step.error_details,
            "current_question": step.current_question,
            "expected_questions": step.expected_questions,
            "sub_steps": step.sub_steps
        }
    
    def submit_step_data(self, workflow_id: str, step_key: str, data: StepDataModel) -> StepStatusUpdate:
        """Submit step data and mark as completed"""
        try:
            # Get or create step
            step = self.step_repo.get_step_by_key(workflow_id, step_key)
            if not step:
                step = self.step_repo.create_step(workflow_id, step_key)
            
            # Update step data and mark as completed
            self.step_repo.update_step_data(step.step_instance_id, data.model_dump())
            self.step_repo.update_step_status(step.step_instance_id, StepStatus.COMPLETED_SUCCESS)
            
            # Update workflow current step
            self._advance_workflow_step(workflow_id, step_key)
            
            self.db.commit()
            
            return StepStatusUpdate(
                step_instance_id=step.step_instance_id,
                status=StepStatus.COMPLETED_SUCCESS,
                message=f"步骤 {step_key} 已完成",
                updated_at=datetime.utcnow()
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def complete_and_advance_step(self, workflow_id: str, step_key: str) -> NextStepInfo:
        """Complete current step and advance to next"""
        try:
            # Get current step
            current_step = self.step_repo.get_step_by_key(workflow_id, step_key)
            if not current_step:
                raise ValueError(f"Step {step_key} not found")
            
            # Mark current step as completed
            self.step_repo.update_step_status(
                current_step.step_instance_id,
                StepStatus.COMPLETED_SUCCESS
            )
            
            # Find next step
            next_step_info = self._get_next_step(workflow_id, step_key)
            
            if next_step_info["next_step_key"]:
                # Activate next step
                next_step = self.step_repo.get_step_by_key(
                    workflow_id, 
                    next_step_info["next_step_key"]
                )
                if next_step:
                    self.step_repo.update_step_status(
                        next_step.step_instance_id,
                        StepStatus.ACTIVE
                    )
                
                # Update workflow current step
                self.instance_repo.update_instance_status(
                    workflow_id,
                    WorkflowStatus.IN_PROGRESS,
                    next_step_info["next_step_key"]
                )
            else:
                # Workflow completed
                self.instance_repo.update_instance_status(
                    workflow_id,
                    WorkflowStatus.COMPLETED
                )
            
            self.db.commit()
            
            return NextStepInfo(
                current_step_completed=True,
                next_step_key=next_step_info["next_step_key"],
                next_step_url=next_step_info.get("next_step_url"),
                workflow_completed=next_step_info["workflow_completed"],
                message=next_step_info["message"]
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def autosave_step_data(self, workflow_id: str, step_key: str, data: StepDataModel) -> AutosaveConfirmation:
        """Autosave step data without changing status"""
        try:
            # Get or create step
            step = self.step_repo.get_step_by_key(workflow_id, step_key)
            if not step:
                step = self.step_repo.create_step(workflow_id, step_key)
            
            # Update step data without changing status
            self.step_repo.update_step_data(step.step_instance_id, data.model_dump())
            
            self.db.commit()
            
            return AutosaveConfirmation(
                step_instance_id=step.step_instance_id,
                saved_at=datetime.utcnow(),
                message="数据已自动保存"
            )
            
        except Exception as e:
            self.db.rollback()
            raise e
    
    def process_form_for_step(self, workflow_id: str, step_key: str, form_html: str, profile_data: dict, profile_dummy_data: dict = None) -> FormProcessResult:
        """Process form HTML for a specific step using LangGraph AI workflow"""
        try:
            # Get workflow instance
            instance = self.instance_repo.get_instance_by_id(workflow_id)
            if not instance:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # 首先进行步骤分析，判断页面属于当前步骤还是下一步骤
            step_analysis = self.form_processor.step_analyzer.analyze_step(form_html, workflow_id, step_key)
            
            # 如果页面属于下一步骤且已完成步骤转换，使用新的步骤键
            actual_step_key = step_key
            if step_analysis.get("should_use_next_step", False) and step_analysis.get("step_transition_completed", False):
                actual_step_key = step_analysis.get("next_step_key", step_key)
                print(f"DEBUG: Using next step {actual_step_key} for form processing")
            
            # Generate simple session ID for tracking
            session_id = f"{workflow_id}_{actual_step_key}"
            
            # Use LangGraph form processor to analyze and generate actions
            result = self.form_processor.process_form(workflow_id, actual_step_key, form_html, profile_data, profile_dummy_data or {})
            
            if result["success"]:
                # Convert actions to FormActionModel list for API response
                actions = []
                for i, action in enumerate(result["actions"]):
                    if action:  # Skip None actions
                        actions.append(FormActionModel(
                            selector=action["selector"],
                            action_type=action["type"],  # 新格式使用type字段
                            value=action.get("value"),
                            order=i
                        ))
                
                # 构建返回结果，包含步骤转换信息
                form_result = FormProcessResult(
                    session_id=session_id,
                    workflow_instance_id=workflow_id,
                    success=True,
                    actions=actions,
                    fields_processed=len(result["actions"]),
                    processing_time_ms=0,  # LangGraph doesn't track time yet
                    # 使用新的简化格式
                    questions=result.get("data"),  # 不再分离返回，data已经包含了合并格式
                    fields_detected=len(result.get("data", [])),
                    processing_metadata={
                        "workflow_id": workflow_id,
                        "original_step_key": step_key,
                        "actual_step_key": actual_step_key,
                        "step_transition_occurred": step_analysis.get("step_transition_completed", False),
                        "step_analysis": {
                            "belongs_to_current_step": step_analysis.get("belongs_to_current_step", True),
                            "belongs_to_next_step": step_analysis.get("belongs_to_next_step", False),
                            "reasoning": step_analysis.get("reasoning", "")
                        },
                        "data_format": "merged_qa",
                        "qa_count": len(result.get("data", []))
                    }
                )
                
                return form_result
            else:
                return FormProcessResult(
                    session_id=session_id,
                    workflow_instance_id=workflow_id,
                    success=False,
                    error_details=result.get("error", "Unknown error occurred"),
                    questions=[],
                    answers=[],
                    processing_metadata={
                        "workflow_id": workflow_id,
                        "original_step_key": step_key,
                        "actual_step_key": actual_step_key,
                        "step_transition_occurred": step_analysis.get("step_transition_completed", False),
                        "error_during_step_analysis": False
                    }
                )
                
        except Exception as e:
            return FormProcessResult(
                session_id="",
                workflow_instance_id=workflow_id,
                success=False,
                error_details=str(e),
                questions=[],
                answers=[],
                processing_metadata={
                    "workflow_id": workflow_id,
                    "original_step_key": step_key,
                    "error_during_step_analysis": True,
                    "error_message": str(e)
                }
            )
    
    def _advance_workflow_step(self, workflow_id: str, current_step_key: str):
        """Advance workflow to next step"""
        # 获取当前步骤
        current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
        if not current_step:
            raise ValueError(f"Current step {current_step_key} not found")
        
        # 获取所有步骤并按顺序排序
        all_steps = self.step_repo.get_workflow_steps(workflow_id)
        sorted_steps = sorted(all_steps, key=lambda s: s.order or 0)
        
        # 找到当前步骤的位置
        current_index = None
        for i, step in enumerate(sorted_steps):
            if step.step_key == current_step_key:
                current_index = i
                break
        
        if current_index is not None and current_index + 1 < len(sorted_steps):
            # 激活下一步
            next_step = sorted_steps[current_index + 1]
            self.step_repo.update_step_status(next_step.step_instance_id, StepStatus.ACTIVE)
            self.instance_repo.update_instance_status(
                workflow_id,
                WorkflowStatus.IN_PROGRESS,
                next_step.step_key
            )
            print(f"✅ 工作流推进到下一步: {next_step.name} ({next_step.step_key})")
        else:
            # 工作流完成
            self.instance_repo.update_instance_status(
                workflow_id,
                WorkflowStatus.COMPLETED
            )
            print("✅ 工作流已完成") 