from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from src.model.workflow_entities import (
    WorkflowDefinition, WorkflowInstance, StepInstance, 
    WorkflowStatus, StepStatus
)
from src.model.workflow_schemas import (
    WorkflowInitiationPayload, WorkflowInstanceSummary, WorkflowInstanceDetail,
    StepDataModel, PersonalDetailsModel, ContactAddressModel
)

class BaseRepository:
    """Base repository class"""
    
    def __init__(self, db_session: Session):
        self.db = db_session

class WorkflowDefinitionRepository(BaseRepository):
    """Workflow definition repository"""
    
    def create_definition(self, name: str, description: str = None, 
                         steps_schema: Dict[str, Any] = None) -> WorkflowDefinition:
        """Create workflow definition"""
        definition = WorkflowDefinition(
            workflow_definition_id=str(uuid.uuid4()),
            name=name,
            description=description,
            step_definitions=steps_schema
        )
        self.db.add(definition)
        self.db.flush()
        return definition
    
    def get_definition_by_id(self, definition_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID"""
        return self.db.query(WorkflowDefinition).filter(
            WorkflowDefinition.workflow_definition_id == definition_id
        ).first()
    
    def get_active_definitions(self) -> List[WorkflowDefinition]:
        """Get all active workflow definitions"""
        return self.db.query(WorkflowDefinition).filter(
            WorkflowDefinition.is_active == True
        ).all()

class WorkflowInstanceRepository(BaseRepository):
    """Workflow instance repository"""
    
    def create_instance(self, user_id: str, workflow_definition_id: str = None) -> WorkflowInstance:
        """Create workflow instance"""
        instance = WorkflowInstance(
            workflow_instance_id=str(uuid.uuid4()),
            user_id=user_id,  # Simple string field, no validation
            workflow_definition_id=workflow_definition_id,
            status=WorkflowStatus.PENDING
        )
        self.db.add(instance)
        self.db.flush()
        return instance
    
    def get_instance_by_id(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance by ID"""
        return self.db.query(WorkflowInstance).filter(
            WorkflowInstance.workflow_instance_id == instance_id
        ).first()
    
    def get_user_instances(self, user_id: str, limit: int = 10) -> List[WorkflowInstance]:
        """Get user's workflow instances"""
        return self.db.query(WorkflowInstance).filter(
            WorkflowInstance.user_id == user_id
        ).order_by(WorkflowInstance.created_at.desc()).limit(limit).all()
    
    def update_instance_status(self, instance_id: str, status: WorkflowStatus, 
                              current_step_key: str = None) -> Optional[WorkflowInstance]:
        """Update workflow instance status"""
        instance = self.get_instance_by_id(instance_id)
        if instance:
            instance.status = status
            instance.updated_at = datetime.utcnow()
            if current_step_key:
                instance.current_step_key = current_step_key
            if status == WorkflowStatus.COMPLETED:
                instance.completed_at = datetime.utcnow()
            return instance
        return None
    
    def pause_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Pause workflow instance"""
        return self.update_instance_status(instance_id, WorkflowStatus.PAUSED)
    
    def resume_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Resume workflow instance"""
        return self.update_instance_status(instance_id, WorkflowStatus.IN_PROGRESS)
    
    def complete_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Complete workflow instance"""
        return self.update_instance_status(instance_id, WorkflowStatus.COMPLETED)
    
    def fail_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Mark workflow instance as failed"""
        return self.update_instance_status(instance_id, WorkflowStatus.FAILED)

class StepInstanceRepository(BaseRepository):
    """Step instance repository"""
    
    def create_step(self, workflow_instance_id: str, step_key: str, name: str = None, order: int = None) -> StepInstance:
        """Create step instance"""
        step = StepInstance(
            step_instance_id=str(uuid.uuid4()),
            workflow_instance_id=workflow_instance_id,
            step_key=step_key,
            name=name,
            order=order,
            status=StepStatus.PENDING
        )
        self.db.add(step)
        self.db.flush()
        return step
    
    def create_steps_from_definition(self, workflow_instance_id: str, step_definitions: List[Dict[str, Any]]) -> List[StepInstance]:
        """Create step instances from workflow definition step_definitions"""
        steps = []
        
        # Sort by order to ensure correct sequence
        sorted_steps = sorted(step_definitions, key=lambda x: x.get('order', 0))
        
        for step_def in sorted_steps:
            step = self.create_step(
                workflow_instance_id=workflow_instance_id,
                step_key=step_def['key'],
                name=step_def.get('name'),
                order=step_def.get('order')
            )
            steps.append(step)
        
        return steps
    
    def get_step_by_id(self, step_id: str) -> Optional[StepInstance]:
        """Get step by ID"""
        return self.db.query(StepInstance).filter(
            StepInstance.step_instance_id == step_id
        ).first()
    
    def get_step_by_key(self, workflow_instance_id: str, step_key: str) -> Optional[StepInstance]:
        """Get step by workflow instance and step key"""
        return self.db.query(StepInstance).filter(
            StepInstance.workflow_instance_id == workflow_instance_id,
            StepInstance.step_key == step_key
        ).first()
    
    def get_workflow_steps(self, workflow_instance_id: str) -> List[StepInstance]:
        """Get all steps for a workflow instance, ordered by order field"""
        return self.db.query(StepInstance).filter(
            StepInstance.workflow_instance_id == workflow_instance_id
        ).order_by(StepInstance.order).all()
    
    def update_step_status(self, step_id: str, status: StepStatus) -> Optional[StepInstance]:
        """Update step status"""
        step = self.get_step_by_id(step_id)
        if step:
            step.status = status
            if status == StepStatus.ACTIVE:
                step.started_at = datetime.utcnow()
            elif status in [StepStatus.COMPLETED_SUCCESS, StepStatus.COMPLETED_ERROR, StepStatus.SKIPPED]:
                step.completed_at = datetime.utcnow()
            return step
        return None
    
    def update_step_data(self, step_id: str, data: Dict[str, Any]) -> Optional[StepInstance]:
        """Update step data"""
        step = self.get_step_by_id(step_id)
        if step:
            step.data = data
            return step
        return None
    
    def set_step_error(self, step_id: str, error_details: str) -> Optional[StepInstance]:
        """Set step error details"""
        step = self.get_step_by_id(step_id)
        if step:
            step.error_details = error_details
            step.status = StepStatus.COMPLETED_ERROR
            step.completed_at = datetime.utcnow()
            return step
        return None 