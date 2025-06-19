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

class WorkflowTypeEnum(str, Enum):
    """Workflow template type enumeration"""
    VISA = "visa"
    PASSPORT = "passport"
    IMMIGRATION = "immigration"
    STUDENT_VISA = "student_visa"
    WORK_PERMIT = "work_permit"
    FAMILY_VISA = "family_visa"

# Base Models
class WorkflowInitiationPayload(BaseModel):
    """Workflow initiation payload"""
    user_id: str = Field(..., description="用户ID")
    case_id: Optional[str] = Field(None, description="案例ID(可选)")
    workflow_definition_id: Optional[str] = Field(None, description="工作流定义ID(可选)")
    initial_data: Optional[Dict[str, Any]] = Field(None, description="初始数据")

class WorkflowInstanceSummary(BaseModel):
    """Workflow instance summary"""
    workflow_instance_id: str = Field(..., description="工作流实例ID")
    user_id: str = Field(..., description="用户ID")
    case_id: Optional[str] = Field(None, description="案例ID")
    status: WorkflowStatusEnum = Field(..., description="状态")
    current_step_key: Optional[str] = Field(None, description="当前步骤键")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")

class WorkflowInstanceDetail(WorkflowInstanceSummary):
    """Workflow instance detail (extends summary)"""
    workflow_definition_id: Optional[str] = Field(None, description="工作流定义ID")
    progress_file_id: Optional[str] = Field(None, description="进度文件ID(外部系统管理)")
    dummy_data_usage: Optional[List[Dict[str, Any]]] = Field(None, description="虚拟数据使用记录JSON数组")
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

# Progress File Models
class ProgressFileUploadRequest(BaseModel):
    """Progress file upload request"""
    file_id: str = Field(..., description="文件ID(外部系统提供)")

class ProgressFileUploadResult(BaseModel):
    """Progress file upload result"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    workflow_instance_id: str = Field(..., description="工作流实例ID")
    progress_file_id: Optional[str] = Field(None, description="已关联的文件ID")
    error_details: Optional[str] = Field(None, description="错误详情")

# WorkflowDefinition Models
class WorkflowDefinitionCreate(BaseModel):
    """Workflow definition creation model"""
    name: str = Field(..., min_length=1, max_length=200, description="工作流名称")
    description: Optional[str] = Field(None, max_length=1000, description="工作流描述")
    type: WorkflowTypeEnum = Field(..., description="模板类型")
    version: str = Field(default="1.0", max_length=20, description="版本号")
    step_definitions: Optional[List[Dict[str, Any]]] = Field(None, description="步骤定义JSON")
    is_active: bool = Field(default=True, description="是否激活")

class WorkflowDefinitionUpdate(BaseModel):
    """Workflow definition update model"""
    name: Optional[str] = Field(None, min_length=1, max_length=200, description="工作流名称")
    description: Optional[str] = Field(None, max_length=1000, description="工作流描述")
    type: Optional[WorkflowTypeEnum] = Field(None, description="模板类型")
    version: Optional[str] = Field(None, max_length=20, description="版本号")
    step_definitions: Optional[List[Dict[str, Any]]] = Field(None, description="步骤定义JSON")
    is_active: Optional[bool] = Field(None, description="是否激活")

class WorkflowDefinitionDetail(BaseModel):
    """Workflow definition detail model"""
    workflow_definition_id: str = Field(..., description="工作流定义ID")
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field(None, description="工作流描述")
    type: WorkflowTypeEnum = Field(..., description="模板类型")
    version: str = Field(..., description="版本号")
    is_active: bool = Field(..., description="是否激活")
    step_definitions: Optional[List[Dict[str, Any]]] = Field(None, description="步骤定义JSON")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")

class WorkflowDefinitionList(BaseModel):
    """Workflow definition list response"""
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页大小")
    items: List[WorkflowDefinitionDetail] = Field(..., description="工作流定义列表")

# Update forward references
WorkflowInstanceDetail.model_rebuild()
StepInstanceDetail.model_rebuild() 