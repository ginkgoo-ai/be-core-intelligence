from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
class WorkflowStatusEnum(str, Enum):
    """Workflow status enumeration"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class StepStatusEnum(str, Enum):
    """Step status enumeration"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_ERROR = "COMPLETED_ERROR"
    SKIPPED = "SKIPPED"

# Base Models
class WorkflowInitiationPayload(BaseModel):
    """Workflow initiation payload"""
    user_id: str = Field(..., description="用户ID")
    workflow_definition_id: Optional[str] = Field(None, description="工作流定义ID(可选)")
    initial_data: Optional[Dict[str, Any]] = Field(None, description="初始数据")

class WorkflowInstanceSummary(BaseModel):
    """Workflow instance summary"""
    workflow_instance_id: str = Field(..., description="工作流实例ID")
    user_id: str = Field(..., description="用户ID")
    status: WorkflowStatusEnum = Field(..., description="状态")
    current_step_key: Optional[str] = Field(None, description="当前步骤键")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

class WorkflowInstanceDetail(WorkflowInstanceSummary):
    """Workflow instance detail (extends summary)"""
    workflow_definition_id: Optional[str] = Field(None, description="工作流定义ID")
    steps: List["StepInstanceDetail"] = Field(default_factory=list, description="步骤列表")

class StepDataModel(BaseModel):
    """Base step data model"""
    step_key: str = Field(..., description="步骤键")
    data: Dict[str, Any] = Field(default_factory=dict, description="步骤数据")

class PersonalDetailsModel(StepDataModel):
    """Personal details step model"""
    step_key: str = Field(default="personal_details", description="步骤键")
    given_name: Optional[str] = Field(None, description="名")
    family_name: Optional[str] = Field(None, description="姓")
    date_of_birth: Optional[str] = Field(None, description="出生日期")
    nationality: Optional[str] = Field(None, description="国籍")
    passport_number: Optional[str] = Field(None, description="护照号码")

class ContactAddressModel(StepDataModel):
    """Contact & Address step model"""
    step_key: str = Field(default="contact_address", description="步骤键")
    email: Optional[str] = Field(None, description="邮箱")
    phone: Optional[str] = Field(None, description="电话")
    address_line1: Optional[str] = Field(None, description="地址行1")
    address_line2: Optional[str] = Field(None, description="地址行2")
    city: Optional[str] = Field(None, description="城市")
    postal_code: Optional[str] = Field(None, description="邮政编码")

class StepInstanceDetail(BaseModel):
    """Step instance detail"""
    step_instance_id: str = Field(..., description="步骤实例ID")
    workflow_instance_id: str = Field(..., description="工作流实例ID")
    step_key: str = Field(..., description="步骤键")
    name: Optional[str] = Field(None, description="步骤名称")
    order: Optional[int] = Field(None, description="步骤顺序")
    status: StepStatusEnum = Field(..., description="状态")
    data: Optional[Dict[str, Any]] = Field(None, description="步骤数据")
    next_step_url: Optional[str] = Field(None, description="下一步URL")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_details: Optional[Dict[str, Any]] = Field(None, description="错误详情")

class StepStatusUpdate(BaseModel):
    """Step status update response"""
    step_instance_id: str = Field(..., description="步骤实例ID")
    status: StepStatusEnum = Field(..., description="更新后状态")
    message: str = Field(..., description="状态消息")
    updated_at: datetime = Field(..., description="更新时间")

class NextStepInfo(BaseModel):
    """Next step information"""
    current_step_completed: bool = Field(..., description="当前步骤是否完成")
    next_step_key: Optional[str] = Field(None, description="下一步骤键")
    next_step_url: Optional[str] = Field(None, description="下一步URL")
    workflow_completed: bool = Field(default=False, description="工作流是否完成")
    message: str = Field(..., description="信息")

class WorkflowStatus(BaseModel):
    """Workflow status response"""
    workflow_instance_id: str = Field(..., description="工作流实例ID")
    status: WorkflowStatusEnum = Field(..., description="状态")
    message: str = Field(..., description="状态消息")
    updated_at: datetime = Field(..., description="更新时间")

class AutosaveConfirmation(BaseModel):
    """Autosave confirmation response"""
    step_instance_id: str = Field(..., description="步骤实例ID")
    saved_at: datetime = Field(..., description="保存时间")
    message: str = Field(default="数据已自动保存", description="确认消息")

# Form-specific models
class FormActionModel(BaseModel):
    """Form action model"""
    selector: str = Field(..., description="CSS选择器")
    action_type: str = Field(..., description="动作类型(input, click, select)")
    value: Optional[str] = Field(None, description="值")
    order: int = Field(..., description="执行顺序")

class FormProcessResult(BaseModel):
    """Form processing result"""
    session_id: str = Field(..., description="会话ID")
    workflow_instance_id: Optional[str] = Field(None, description="工作流实例ID")
    success: bool = Field(..., description="是否成功")
    actions: List[FormActionModel] = Field(default_factory=list, description="表单动作列表")
    fields_processed: int = Field(default=0, description="处理的字段数")
    processing_time_ms: int = Field(default=0, description="处理时间(毫秒)")
    error_details: Optional[str] = Field(None, description="错误详情")
    questions: List[Dict[str, Any]] = Field(default_factory=list, description="HTML页面分析出的问题")
    answers: List[Dict[str, Any]] = Field(default_factory=list, description="AI生成的答案")
    fields_detected: int = Field(default=0, description="检测到的字段数")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="处理元数据")

class FormDataResult(BaseModel):
    """Form data result in new merged format"""
    workflow_id: str = Field(..., description="工作流ID")
    step_key: str = Field(..., description="步骤键")
    success: bool = Field(..., description="是否成功")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="合并的问题答案数据")
    actions: List[Dict[str, str]] = Field(default_factory=list, description="Google插件格式的动作")
    error_details: Optional[str] = Field(None, description="错误详情")

# Update forward references
WorkflowInstanceDetail.model_rebuild()
StepInstanceDetail.model_rebuild() 