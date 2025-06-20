from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from src.database.database_config import get_db_session
from src.business.workflow_service import WorkflowService, StepService, WorkflowDefinitionService
from src.model.workflow_schemas import (
    WorkflowInitiationPayload, WorkflowInstanceSummary, WorkflowInstanceDetail,
    WorkflowStatus, StepDataModel, PersonalDetailsModel, ContactAddressModel,
    StepStatusUpdate, NextStepInfo, AutosaveConfirmation, FormProcessResult,
    FormDataResult, ProgressFileUploadRequest, ProgressFileUploadResult,
    WorkflowDefinitionCreate, WorkflowDefinitionUpdate, WorkflowDefinitionDetail,
    WorkflowDefinitionList, WorkflowTypeEnum
)
from src.business.langgraph_form_processor import LangGraphFormProcessor
from src.model.workflow_entities import WorkflowInstance, StepInstance, WorkflowStatus, StepStatus

# Create router
workflow_router = APIRouter(prefix="/workflows", tags=["workflow"])

class FormProcessRequest(BaseModel):
    """Request model for form processing"""
    form_html: str
    profile_data: dict
    profile_dummy_data: Optional[dict] = None  # Add optional dummy data field

class DummyDataUsageResult(BaseModel):
    """Result model for dummy data usage tracking"""
    workflow_id: str = Field(..., description="工作流ID")
    total_dummy_records: int = Field(..., description="虚拟数据记录总数")
    ai_generated_count: int = Field(..., description="AI生成的虚拟数据数量")
    provided_dummy_count: int = Field(..., description="提供的虚拟数据数量")
    dummy_usage: List[Dict[str, Any]] = Field(default_factory=list, description="虚拟数据使用详情")
    steps_with_dummy: List[str] = Field(default_factory=list, description="使用了虚拟数据的步骤")

def get_workflow_service(db: Session = Depends(get_db_session)) -> WorkflowService:
    """Get workflow service instance"""
    return WorkflowService(db)

def get_step_service(db: Session = Depends(get_db_session)) -> StepService:
    """Get step service instance"""
    return StepService(db)

@workflow_router.post("/", response_model=WorkflowInstanceSummary)
async def create_workflow(
    payload: WorkflowInitiationPayload,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Create a new workflow instance
    
    创建新的工作流实例，启动12步签证申请流程
    """
    try:
        return service.create_workflow(payload)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建工作流失败: {str(e)}"
        )

@workflow_router.get("/user/{user_id}", response_model=List[WorkflowInstanceSummary])
async def get_user_workflows(
    user_id: str,
    limit: int = 50,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Get user's workflow instances list
    
    获取指定用户的所有工作流实例列表，按创建时间倒序排列
    """
    try:
        return service.get_user_workflows(user_id, limit)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户工作流列表失败: {str(e)}"
        )

@workflow_router.get("/case/{case_id}", response_model=List[WorkflowInstanceSummary])
async def get_case_workflows(
    case_id: str,
    limit: int = 50,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Get workflow instances by case ID
    
    获取指定案例的所有工作流实例列表，按创建时间倒序排列
    """
    try:
        return service.get_case_workflows(case_id, limit)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取案例工作流列表失败: {str(e)}"
        )

@workflow_router.get("/user/{user_id}/case/{case_id}", response_model=List[WorkflowInstanceSummary])
async def get_user_case_workflows(
    user_id: str,
    case_id: str,
    limit: int = 50,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Get workflow instances by user ID and case ID
    
    获取指定用户在指定案例下的所有工作流实例列表，按创建时间倒序排列
    """
    try:
        return service.get_user_case_workflows(user_id, case_id, limit)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户案例工作流列表失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/", response_model=WorkflowInstanceDetail)
async def get_workflow_status(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Get workflow status and details
    
    获取工作流状态和详细信息，包括所有步骤的执行情况
    """
    try:
        return service.get_workflow_status(workflow_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取工作流状态失败: {str(e)}"
        )

@workflow_router.post("/{workflow_id}/pause", response_model=WorkflowStatus)
async def pause_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Pause workflow execution
    
    暂停工作流执行，用户可以稍后恢复
    """
    try:
        return service.pause_workflow(workflow_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"暂停工作流失败: {str(e)}"
        )

@workflow_router.post("/{workflow_id}/resume", response_model=WorkflowStatus)
async def resume_workflow(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Resume workflow execution
    
    恢复已暂停的工作流执行
    """
    try:
        return service.resume_workflow(workflow_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"恢复工作流失败: {str(e)}"
        )

# Step management endpoints
@workflow_router.get("/{workflow_id}/steps/{step_key}/", response_model=StepDataModel)
async def get_step_data(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get step data
    
    获取指定步骤的数据，支持不同步骤类型的专用模型
    """
    try:
        return service.get_step_data(workflow_id, step_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取步骤数据失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/raw", response_model=Dict[str, Any])
async def get_step_raw_data(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get step raw JSON data from database
    
    获取步骤在数据库中存储的原始JSON数据，用于调试和高级用途
    """
    try:
        return service.get_step_raw_data(workflow_id, step_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取步骤原始数据失败: {str(e)}"
        )

@workflow_router.patch("/{workflow_id}/steps/{step_key}/", response_model=StepStatusUpdate)
async def submit_step_data(
    workflow_id: str,
    step_key: str,
    data: StepDataModel,
    service: StepService = Depends(get_step_service)
):
    """Submit step data
    
    提交步骤数据并标记为完成，自动推进到下一步
    """
    try:
        return service.submit_step_data(workflow_id, step_key, data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交步骤数据失败: {str(e)}"
        )

@workflow_router.patch("/{workflow_id}/steps/{step_key}/complete", response_model=NextStepInfo)
async def complete_and_advance_step(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Complete current step and advance to next
    
    完成当前步骤并推进到下一步，返回下一步信息
    """
    try:
        return service.complete_and_advance_step(workflow_id, step_key)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"完成步骤失败: {str(e)}"
        )

@workflow_router.patch("/{workflow_id}/steps/{step_key}/autosave", response_model=AutosaveConfirmation)
async def autosave_step_data(
    workflow_id: str,
    step_key: str,
    data: StepDataModel,
    service: StepService = Depends(get_step_service)
):
    """Autosave step data
    
    自动保存步骤数据，不改变步骤状态，用于防止数据丢失
    """
    try:
        return service.autosave_step_data(workflow_id, step_key, data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"自动保存失败: {str(e)}"
        )

@workflow_router.post("/{workflow_id}/steps/{step_key}/process-form", response_model=FormProcessResult)
async def process_form_for_step(
    workflow_id: str,
    step_key: str,
    request: FormProcessRequest,
    service: StepService = Depends(get_step_service)
):
    """Process form HTML for a specific step
    
    使用AI分析表单HTML并生成自动填写动作，所有分析结果保存在StepInstance中
    
    Args:
        workflow_id: 工作流实例ID
        step_key: 步骤键  
        request: 包含form_html, profile_data和profile_dummy_data的请求体
    """
    try:
        return service.process_form_for_step(
            workflow_id, step_key, request.form_html, request.profile_data, request.profile_dummy_data
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理表单失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/analysis", response_model=Dict[str, Any])
async def get_step_analysis_data(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get detailed analysis data for a step
    
    获取步骤的详细分析数据，包括HTML解析结果、AI问题和答案
    """
    try:
        # Get step data
        step_data = service.get_step_data(workflow_id, step_key)
        
        # Extract analysis data from step data
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        # 从新的数据结构中提取分析信息
        form_data = analysis_data.get("form_data", [])
        actions = analysis_data.get("actions", [])
        questions = analysis_data.get("questions", [])
        metadata = analysis_data.get("metadata", {})
        history = analysis_data.get("history", [])
        
        return {
            "workflow_id": workflow_id,
            "step_key": step_key,
            "analysis_data": analysis_data,
            "has_form_data": len(form_data) > 0,
            "has_actions": len(actions) > 0,
            "has_questions": len(questions) > 0,
            "has_metadata": bool(metadata),
            "has_history": len(history) > 0,
            "form_data_count": len(form_data),
            "question_count": len(questions),
            "action_count": len(actions),
            "history_count": len(history),
            "last_processed": metadata.get("processed_at", ""),
            "success": metadata.get("success", False)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取分析数据失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/questions", response_model=List[Dict[str, Any]])
async def get_step_questions(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get AI-generated questions for form fields
    
    获取为表单字段生成的AI问题列表
    """
    try:
        step_data = service.get_step_data(workflow_id, step_key)
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        # 从新的数据结构中获取问题数据
        questions = analysis_data.get("questions", [])
        
        return questions
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取问题列表失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/answers", response_model=List[Dict[str, Any]])
async def get_step_answers(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get AI-generated answers for form fields
    
    获取AI为表单字段生成的答案列表（从合并的问答数据中提取）
    """
    try:
        step_data = service.get_step_data(workflow_id, step_key)
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        # 从新的数据结构中的 form_data 提取答案信息
        form_data = analysis_data.get("form_data", [])
        
        # 提取所有答案数据
        answers = []
        for item in form_data:
            question_data = item.get("question", {})
            answer_data = question_data.get("answer", {})
            
            # 构建答案对象
            answer_item = {
                "question_name": question_data.get("data", {}).get("name", ""),
                "answer_type": answer_data.get("type", ""),
                "selector": answer_data.get("selector", ""),
                "data": answer_data.get("data", []),
                "interrupt": 1 if question_data.get("type") == "interrupt" else 0
            }
            
            # 添加元数据（如果存在）
            if "_metadata" in item:
                metadata = item["_metadata"]
                answer_item.update({
                    "field_name": metadata.get("field_name", ""),
                    "field_type": metadata.get("field_type", ""),
                    "confidence": metadata.get("confidence", 0),
                    "reasoning": metadata.get("reasoning", ""),
                    "needs_intervention": metadata.get("needs_intervention", False)
                })
            
            answers.append(answer_item)
        
        return answers
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取答案列表失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/actions", response_model=List[Dict[str, Any]])
async def get_step_actions(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get generated form actions for a step
    
    获取为步骤生成的表单操作列表
    """
    try:
        step_data = service.get_step_data(workflow_id, step_key)
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        # 从新的数据结构中获取动作数据
        actions = analysis_data.get("actions", [])
        
        return actions
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取操作列表失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/steps/{step_key}/data", response_model=FormDataResult)
async def get_step_merged_data(
    workflow_id: str,
    step_key: str,
    service: StepService = Depends(get_step_service)
):
    """Get step data in merged Q&A format
    
    获取步骤数据，返回合并的问题答案格式
    格式: [{"question":{"name":"..."},"answer":[{"value":"china", "check":1},{"value":"japan"}]}]
    """
    try:
        # Get step data
        step_data = service.get_step_data(workflow_id, step_key)
        
        # Extract analysis data from step data
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        # Get the merged data and actions from the new structure
        merged_data = analysis_data.get("form_data", [])  # 使用新的 form_data 字段
        actions = analysis_data.get("actions", [])  # 使用新的 actions 字段
        
        # Convert actions to Google plugin format
        google_actions = []
        for action in actions:
            google_action = {
                "selector": action.get("selector", ""),
                "type": action.get("type", "")  # 新格式直接使用 type 字段
            }
            if action.get("value"):
                google_action["value"] = action.get("value")
            google_actions.append(google_action)
        
        return FormDataResult(
            workflow_id=workflow_id,
            step_key=step_key,
            success=True,
            data=merged_data,
            actions=google_actions
        )
        
    except ValueError as e:
        return FormDataResult(
            workflow_id=workflow_id,
            step_key=step_key,
            success=False,
            data=[],
            actions=[],
            error_details=str(e)
        )
    except Exception as e:
        return FormDataResult(
            workflow_id=workflow_id,
            step_key=step_key,
            success=False,
            data=[],
            actions=[],
            error_details=f"获取数据失败: {str(e)}"
        )

@workflow_router.post("/{workflow_id}/process-form")
async def process_form(
    workflow_id: str,
    form_data: Dict[str, Any],
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    处理表单数据 - 使用智能步骤分析
    Args:
        workflow_id: 工作流ID
        form_data: 表单数据，包含 form_html, profile_data 和 profile_dummy_data
    Returns:
        处理结果
    """
    try:
        # 获取工作流实例
        workflow_instance = db.query(WorkflowInstance).filter(
            WorkflowInstance.workflow_instance_id == workflow_id
        ).first()
        
        if not workflow_instance:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # 获取当前步骤
        current_step = None
        if workflow_instance.current_step_key:
            current_step = db.query(StepInstance).filter(
                StepInstance.workflow_instance_id == workflow_id,
                StepInstance.step_key == workflow_instance.current_step_key
            ).first()
        
        # 如果没有当前步骤，获取第一个待处理的步骤
        if not current_step:
            current_step = db.query(StepInstance).filter(
                StepInstance.workflow_instance_id == workflow_id,
                StepInstance.status == StepStatus.PENDING
            ).order_by(StepInstance.order).first()
            
            if not current_step:
                raise HTTPException(status_code=404, detail="No pending steps found")
            
            # 更新工作流当前步骤
            workflow_instance.current_step_key = current_step.step_key
            db.commit()
        
        # 使用 StepService 进行智能表单处理（包含步骤分析和切换逻辑）
        step_service = StepService(db)
        form_result = step_service.process_form_for_step(
            workflow_id=workflow_id,
            step_key=current_step.step_key,
            form_html=form_data.get("form_html", ""),
            profile_data=form_data.get("profile_data", {}),
            profile_dummy_data=form_data.get("profile_dummy_data", {})
        )
        
        # 转换 FormProcessResult 为 API 响应格式
        result = {
            "success": form_result.success,
            "data": form_result.questions,  # 使用 questions 字段（包含合并的问答数据）
            "actions": [
                {
                    "selector": action.selector,
                    "type": action.action_type,
                    "value": action.value
                } for action in form_result.actions
            ],
            "processing_metadata": form_result.processing_metadata
        }
        
        # 如果有错误，添加错误信息
        if not form_result.success:
            result["error"] = form_result.error_details
        
        # 检查是否发生了步骤转换
        metadata = form_result.processing_metadata
        if metadata.get("step_transition_occurred", False):
            result["step_transition"] = {
                "occurred": True,
                "original_step": metadata.get("original_step_key"),
                "new_step": metadata.get("actual_step_key"),
                "reasoning": metadata.get("step_analysis", {}).get("reasoning", "")
            }
            
            # 更新工作流实例的当前步骤（如果步骤转换成功）
            new_step_key = metadata.get("actual_step_key")
            if new_step_key and new_step_key != current_step.step_key:
                workflow_instance.current_step_key = new_step_key
                db.commit()
        
        return result
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.get("/{workflow_id}/current-step")
async def get_current_step(
    workflow_id: str,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    获取当前步骤信息
    Args:
        workflow_id: 工作流ID
    Returns:
        当前步骤信息
    """
    try:
        # 获取工作流实例
        workflow_instance = db.query(WorkflowInstance).filter(
            WorkflowInstance.workflow_instance_id == workflow_id
        ).first()
        
        if not workflow_instance:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # 获取当前步骤
        current_step = None
        if workflow_instance.current_step_key:
            current_step = db.query(StepInstance).filter(
                StepInstance.workflow_instance_id == workflow_id,
                StepInstance.step_key == workflow_instance.current_step_key
            ).first()
        
        # 如果没有当前步骤，获取第一个待处理的步骤
        if not current_step:
            current_step = db.query(StepInstance).filter(
                StepInstance.workflow_instance_id == workflow_id,
                StepInstance.status == StepStatus.PENDING
            ).order_by(StepInstance.order).first()
            
            if not current_step:
                raise HTTPException(status_code=404, detail="No pending steps found")
            
            # 更新工作流当前步骤
            workflow_instance.current_step_key = current_step.step_key
            current_step.status = StepStatus.ACTIVE
            current_step.started_at = datetime.utcnow()
            db.commit()
        
        return {
            "step_key": current_step.step_key,
            "name": current_step.name,
            "status": current_step.status.value,
            "current_question": current_step.current_question,
            "expected_questions": current_step.expected_questions,
            "sub_steps": current_step.sub_steps,
            "data": current_step.data,
            "error_details": current_step.error_details
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@workflow_router.post("/{workflow_id}/upload-progress-file", response_model=ProgressFileUploadResult)
async def upload_progress_file(
    workflow_id: str,
    request: ProgressFileUploadRequest,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Upload progress file for workflow instance
    
    为工作流实例上传进度文件，只需要提供file_id
    """
    try:
        return service.upload_progress_file(workflow_id, request.file_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"上传进度文件失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/progress-file", response_model=Dict[str, Any])
async def get_progress_file(
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service)
):
    """Get progress file information for workflow instance
    
    获取工作流实例的进度文件信息
    """
    try:
        return service.get_progress_file(workflow_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取进度文件信息失败: {str(e)}"
        )

# WorkflowDefinition CRUD APIs
@workflow_router.post("/definitions", response_model=WorkflowDefinitionDetail, status_code=status.HTTP_201_CREATED)
async def create_workflow_definition(
    definition_data: WorkflowDefinitionCreate,
    db: Session = Depends(get_db_session)
):
    """Create a new workflow definition"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.create_definition(definition_data)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建工作流定义失败: {str(e)}"
        )

@workflow_router.get("/definitions/{definition_id}", response_model=WorkflowDefinitionDetail)
async def get_workflow_definition(
    definition_id: str,
    db: Session = Depends(get_db_session)
):
    """Get workflow definition by ID"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.get_definition_by_id(definition_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"工作流定义未找到: {definition_id}"
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取工作流定义失败: {str(e)}"
        )

@workflow_router.put("/definitions/{definition_id}", response_model=WorkflowDefinitionDetail)
async def update_workflow_definition(
    definition_id: str,
    update_data: WorkflowDefinitionUpdate,
    db: Session = Depends(get_db_session)
):
    """Update workflow definition"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.update_definition(definition_id, update_data)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"工作流定义未找到: {definition_id}"
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新工作流定义失败: {str(e)}"
        )

@workflow_router.delete("/definitions/{definition_id}")
async def delete_workflow_definition(
    definition_id: str,
    hard_delete: bool = False,
    db: Session = Depends(get_db_session)
):
    """Delete workflow definition (soft delete by default, hard delete if hard_delete=true)"""
    try:
        service = WorkflowDefinitionService(db)
        success = service.delete_definition(definition_id, hard_delete)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"工作流定义未找到: {definition_id}"
            )
        
        delete_type = "硬删除" if hard_delete else "软删除"
        return {
            "success": True,
            "message": f"工作流定义{delete_type}成功",
            "definition_id": definition_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除工作流定义失败: {str(e)}"
        )

@workflow_router.get("/definitions", response_model=WorkflowDefinitionList)
async def get_workflow_definitions(
    page: int = 1,
    page_size: int = 10,
    workflow_type: Optional[WorkflowTypeEnum] = None,
    is_active: Optional[bool] = None,
    search_term: Optional[str] = None,
    db: Session = Depends(get_db_session)
):
    """Get paginated workflow definitions with filters"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.get_definitions_list(
            page=page,
            page_size=page_size,
            workflow_type=workflow_type.value if workflow_type else None,
            is_active=is_active,
            search_term=search_term
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取工作流定义列表失败: {str(e)}"
        )

@workflow_router.get("/definitions/by-type/{workflow_type}", response_model=List[WorkflowDefinitionDetail])
async def get_workflow_definitions_by_type(
    workflow_type: WorkflowTypeEnum,
    is_active: bool = True,
    db: Session = Depends(get_db_session)
):
    """Get workflow definitions by type"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.get_definitions_by_type(workflow_type.value, is_active)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"按类型获取工作流定义失败: {str(e)}"
        )

@workflow_router.get("/definitions/all", response_model=List[WorkflowDefinitionDetail])
async def get_all_workflow_definitions(
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db_session)
):
    """Get all workflow definitions"""
    try:
        service = WorkflowDefinitionService(db)
        result = service.get_all_definitions(is_active)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取所有工作流定义失败: {str(e)}"
        )

@workflow_router.get("/{workflow_id}/dummy-data-usage", response_model=DummyDataUsageResult)
async def get_dummy_data_usage(
    workflow_id: str,
    db: Session = Depends(get_db_session)
):
    """Get dummy data usage statistics for a workflow
    
    获取工作流中虚拟数据的使用情况统计
    """
    try:
        # Get workflow instance
        workflow_instance = db.query(WorkflowInstance).filter(
            WorkflowInstance.workflow_instance_id == workflow_id
        ).first()
        
        if not workflow_instance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found"
            )
        
        dummy_usage = workflow_instance.dummy_data_usage or []
        
        # Analyze dummy data usage
        ai_generated_count = 0
        provided_dummy_count = 0
        steps_with_dummy = set()
        
        for record in dummy_usage:
            source = record.get("source", "unknown")
            step_key = record.get("step_key", "")
            
            if source == "ai_generated":
                ai_generated_count += 1
            elif source in ["profile_dummy_data", "provided"]:
                provided_dummy_count += 1
            
            if step_key:
                steps_with_dummy.add(step_key)
        
        return DummyDataUsageResult(
            workflow_id=workflow_id,
            total_dummy_records=len(dummy_usage),
            ai_generated_count=ai_generated_count,
            provided_dummy_count=provided_dummy_count,
            dummy_usage=dummy_usage,
            steps_with_dummy=list(steps_with_dummy)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取虚拟数据使用记录失败: {str(e)}"
        ) 