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
    
    def create_definition(self, name: str, description: str = None, 
                         step_definitions: List[Dict[str, Any]] = None) -> WorkflowDefinition:
        """Create workflow definition"""
        definition = WorkflowDefinition(
            workflow_definition_id=str(uuid.uuid4()),
            name=name,
            description=description,
            step_definitions=step_definitions
        )
        self.db.add(definition)
        self.db.flush()
        return definition
    
    def create_definition_full(self, name: str, description: str = None, 
                              workflow_type: str = "visa", version: str = "1.0",
                              step_definitions: List[Dict[str, Any]] = None,
                              is_active: bool = True) -> WorkflowDefinition:
        """Create workflow definition with full parameters"""
        definition = WorkflowDefinition(
            workflow_definition_id=str(uuid.uuid4()),
            name=name,
            description=description,
            type=workflow_type,
            version=version,
            step_definitions=step_definitions,
            is_active=is_active
        )
        self.db.add(definition)
        self.db.flush()
        return definition
    
    def update_definition(self, definition_id: str, **kwargs) -> Optional[WorkflowDefinition]:
        """Update workflow definition"""
        definition = self.get_definition_by_id(definition_id)
        if not definition:
            return None
        
        # Update allowed fields
        allowed_fields = ['name', 'description', 'type', 'version', 'step_definitions', 'is_active']
        for field, value in kwargs.items():
            if field in allowed_fields and value is not None:
                setattr(definition, field, value)
        
        definition.updated_at = datetime.utcnow()
        self.db.flush()
        return definition
    
    def delete_definition(self, definition_id: str) -> bool:
        """Delete workflow definition (soft delete by setting is_active=False)"""
        definition = self.get_definition_by_id(definition_id)
        if not definition:
            return False
        
        definition.is_active = False
        definition.updated_at = datetime.utcnow()
        self.db.flush()
        return True
    
    def hard_delete_definition(self, definition_id: str) -> bool:
        """Hard delete workflow definition"""
        definition = self.get_definition_by_id(definition_id)
        if not definition:
            return False
        
        self.db.delete(definition)
        self.db.flush()
        return True
    
    def get_definitions_by_type(self, workflow_type: str, is_active: bool = True) -> List[WorkflowDefinition]:
        """Get workflow definitions by type"""
        query = self.db.query(WorkflowDefinition).filter(
            WorkflowDefinition.type == workflow_type
        )
        if is_active is not None:
            query = query.filter(WorkflowDefinition.is_active == is_active)
        return query.all()
    
    def get_definitions_paginated(self, page: int = 1, page_size: int = 10, 
                                 workflow_type: str = None, is_active: bool = None,
                                 search_term: str = None) -> Dict[str, Any]:
        """Get paginated workflow definitions with filters"""
        query = self.db.query(WorkflowDefinition)
        
        # Apply filters
        if workflow_type:
            query = query.filter(WorkflowDefinition.type == workflow_type)
        if is_active is not None:
            query = query.filter(WorkflowDefinition.is_active == is_active)
        if search_term:
            query = query.filter(
                WorkflowDefinition.name.ilike(f"%{search_term}%") |
                WorkflowDefinition.description.ilike(f"%{search_term}%")
            )
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        items = query.order_by(WorkflowDefinition.created_at.desc()).offset(offset).limit(page_size).all()
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": items
        }
    
    def get_all_definitions(self, is_active: bool = None) -> List[WorkflowDefinition]:
        """Get all workflow definitions"""
        query = self.db.query(WorkflowDefinition)
        if is_active is not None:
            query = query.filter(WorkflowDefinition.is_active == is_active)
        return query.order_by(WorkflowDefinition.created_at.desc()).all()

class WorkflowInstanceRepository(BaseRepository):
    """Workflow instance repository"""
    
    def create_instance(self, user_id: str, case_id: str = None, workflow_definition_id: str = None) -> WorkflowInstance:
        """Create workflow instance"""
        instance = WorkflowInstance(
            workflow_instance_id=str(uuid.uuid4()),
            user_id=user_id,  # Simple string field, no validation
            case_id=case_id,
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
    
    def get_case_instances(self, case_id: str, limit: int = 50) -> List[WorkflowInstance]:
        """Get workflow instances by case ID"""
        return self.db.query(WorkflowInstance).filter(
            WorkflowInstance.case_id == case_id
        ).order_by(WorkflowInstance.created_at.desc()).limit(limit).all()
    
    def get_user_case_instances(self, user_id: str, case_id: str, limit: int = 50) -> List[WorkflowInstance]:
        """Get workflow instances by user ID and case ID"""
        return self.db.query(WorkflowInstance).filter(
            WorkflowInstance.user_id == user_id,
            WorkflowInstance.case_id == case_id
        ).order_by(WorkflowInstance.created_at.desc()).limit(limit).all()
    
    def update_instance_status(self, instance_id: str, status: WorkflowStatus, 
                              current_step_key: str = None) -> Optional[WorkflowInstance]:
        """Update workflow instance status"""
        instance = self.get_instance_by_id(instance_id)
        if instance:
            # Only update status if it's not None (allows updating only current_step_key)
            if status is not None:
                instance.status = status
                if status == WorkflowStatus.COMPLETED:
                    instance.completed_at = datetime.utcnow()
            
            instance.updated_at = datetime.utcnow()
            if current_step_key:
                instance.current_step_key = current_step_key
            return instance
        return None
    
    def update_progress_file_id(self, instance_id: str, file_id: str) -> Optional[WorkflowInstance]:
        """Update workflow instance progress file ID"""
        instance = self.get_instance_by_id(instance_id)
        if instance:
            instance.progress_file_id = file_id
            instance.updated_at = datetime.utcnow()
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

class StepInstanceRepository(BaseRepository):
    """Step instance repository"""
    
    def create_step(self, workflow_instance_id: str, step_key: str, name: str = None, 
                   order: int = None) -> StepInstance:
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
            step.updated_at = datetime.utcnow()
            
            if status == StepStatus.ACTIVE:
                step.started_at = datetime.utcnow()
            elif status in [StepStatus.COMPLETED_SUCCESS, StepStatus.COMPLETED_ERROR]:
                step.completed_at = datetime.utcnow()
            
            return step
        return None
    
    def update_step_data(self, step_id: str, data: Dict[str, Any]) -> Optional[StepInstance]:
        """Update step data"""
        step = self.get_step_by_id(step_id)
        if step:
            step.data = data
            step.updated_at = datetime.utcnow()
            return step
        return None
    
    def set_step_error(self, step_id: str, error_details: str) -> Optional[StepInstance]:
        """Set step error details"""
        step = self.get_step_by_id(step_id)
        if step:
            step.error_details = error_details
            step.status = StepStatus.COMPLETED_ERROR
            step.updated_at = datetime.utcnow()
            step.completed_at = datetime.utcnow()
            return step
        return None 