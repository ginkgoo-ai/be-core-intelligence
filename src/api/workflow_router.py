from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel

from src.database.database_config import get_db_session
from src.business.workflow_service import WorkflowService, StepService
from src.model.workflow_schemas import (
    WorkflowInitiationPayload, WorkflowInstanceSummary, WorkflowInstanceDetail,
    WorkflowStatus, StepDataModel, PersonalDetailsModel, ContactAddressModel,
    StepStatusUpdate, NextStepInfo, AutosaveConfirmation, FormProcessResult,
    FormDataResult
)

# Create router
workflow_router = APIRouter(prefix="/workflows", tags=["workflows"])

class FormProcessRequest(BaseModel):
    """Request model for form processing"""
    form_html: str
    profile_data: dict

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
        request: 包含form_html和profile_data的请求体
    """
    try:
        return service.process_form_for_step(
            workflow_id, step_key, request.form_html, request.profile_data
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
        
        return {
            "workflow_id": workflow_id,
            "step_key": step_key,
            "analysis_data": analysis_data,
            "has_html_analysis": "html_analysis" in analysis_data,
            "has_ai_processing": "ai_processing" in analysis_data,
            "has_form_actions": "form_actions" in analysis_data,
            "field_count": analysis_data.get("html_analysis", {}).get("field_count", 0),
            "question_count": analysis_data.get("ai_processing", {}).get("question_count", 0),
            "answer_count": analysis_data.get("ai_processing", {}).get("answer_count", 0),
            "action_count": len(analysis_data.get("form_actions", []))
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
        
        questions = analysis_data.get("ai_processing", {}).get("questions", [])
        
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
    
    获取AI为表单字段生成的答案列表
    """
    try:
        step_data = service.get_step_data(workflow_id, step_key)
        analysis_data = step_data.data if hasattr(step_data, 'data') else {}
        
        answers = analysis_data.get("ai_processing", {}).get("answers", [])
        
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
        
        actions = analysis_data.get("form_actions", [])
        
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
        
        # Get the merged data and actions
        merged_data = analysis_data.get("data", [])
        actions = analysis_data.get("actions", [])
        
        # Convert actions to Google plugin format
        google_actions = []
        for action in actions:
            google_action = {
                "selector": action.get("selector", ""),
                "type": action.get("action_type", "")
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
            error_message=str(e)
        )
    except Exception as e:
        return FormDataResult(
            workflow_id=workflow_id,
            step_key=step_key,
            success=False,
            data=[],
            actions=[],
            error_message=f"获取数据失败: {str(e)}"
        ) 