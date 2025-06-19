import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum as SQLEnum, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"  
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class StepStatus(Enum):
    """Step status enumeration"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_ERROR = "COMPLETED_ERROR"
    SKIPPED = "SKIPPED"

class WorkflowDefinition(Base):
    """Workflow definition entity - 工作流定义表"""
    __tablename__ = 'workflow_definitions'
    __table_args__ = {'schema': 'workflow'}
    
    workflow_definition_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, comment="工作流名称")
    description = Column(Text, nullable=True, comment="工作流描述")
    version = Column(String(20), default="1.0", comment="版本号")
    type = Column(String(50), nullable=False, default="visa", comment="模板类型(visa,passport,immigration等)")
    is_active = Column(Boolean, default=True, comment="是否激活")
    step_definitions = Column(JSON, nullable=True, comment="步骤定义JSON")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    # Relationships
    workflow_instances = relationship("WorkflowInstance", back_populates="workflow_definition")

class WorkflowInstance(Base):
    """Workflow instance entity - 工作流实例表"""
    __tablename__ = 'workflow_instances'
    __table_args__ = {'schema': 'workflow'}
    
    workflow_instance_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(50), nullable=False, comment="用户ID")  # Removed foreign key constraint
    case_id = Column(String(50), nullable=True, comment="案例ID(一个案例可以包含多个工作流实例)")
    workflow_definition_id = Column(String(50), ForeignKey('workflow.workflow_definitions.workflow_definition_id'), 
                                   nullable=True, comment="工作流定义XID(外键,可选)")
    status = Column(SQLEnum(WorkflowStatus), default=WorkflowStatus.PENDING, 
                   comment="状态(PENDING,IN_PROGRESS,PAUSED,COMPLETED,FAILED)")
    current_step_key = Column(String(100), nullable=True, comment="当前步骤键")
    progress_file_id = Column(String(100), nullable=True, comment="进度文件ID(外部系统管理)")
    dummy_data_usage = Column(JSON, nullable=True, comment="虚拟数据使用记录JSON数组: [{'processed_at':'2024-01-01T12:00:00','step_key':'personal_details','question':'请选择您的国籍','answer':'i am dummy data'}]")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    
    # Relationships (removed user relationship)
    workflow_definition = relationship("WorkflowDefinition", back_populates="workflow_instances")
    step_instances = relationship("StepInstance", back_populates="workflow_instance")

class StepInstance(Base):
    """Step instance entity - 步骤实例表"""
    __tablename__ = 'step_instances'
    __table_args__ = {'schema': 'workflow'}
    
    step_instance_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_instance_id = Column(String(50), ForeignKey('workflow.workflow_instances.workflow_instance_id'), 
                                 nullable=False, comment="工作流实例ID(外键)")
    step_key = Column(String(100), nullable=False, comment="步骤键")
    name = Column(String(200), nullable=True, comment="步骤名称")
    order = Column(Integer, nullable=True, comment="步骤顺序")
    status = Column(SQLEnum(StepStatus), default=StepStatus.PENDING, 
                   comment="状态(PENDING,ACTIVE,COMPLETED_SUCCESS,COMPLETED_ERROR,SKIPPED)")
    data = Column(JSON, nullable=True, comment="步骤数据JSON(包含AI分析结果)")
    next_step_url = Column(String(500), nullable=True, comment="下一步URL")
    started_at = Column(DateTime, nullable=True, comment="开始时间")
    completed_at = Column(DateTime, nullable=True, comment="完成时间")
    error_details = Column(Text, nullable=True, comment="错误详情")
    current_question = Column(String, comment="当前页面要回答的问题")
    expected_questions = Column(JSON, comment="步骤预期要回答的问题列表")
    sub_steps = Column(JSON, comment="子步骤信息")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    # Relationships
    workflow_instance = relationship("WorkflowInstance", back_populates="step_instances")

