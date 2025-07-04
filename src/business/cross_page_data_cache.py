from datetime import datetime, timedelta
from typing import Dict, List, Any

from sqlalchemy.orm import Session

from src.database.workflow_repositories import StepInstanceRepository
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


class CrossPageDataCache:
    """
    Cross-page data cache manager
    Used to solve cross-page data association problems in UK visa application forms
    
    Core functionalities:
    1. Cache processed data from each page
    2. Provide cross-page data query interface
    3. Intelligent data mapping and association analysis
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.step_repo = StepInstanceRepository(db_session)
        self.cache_ttl = 3600  # 1 hour cache expiry time
    
    def cache_page_data(self, workflow_id: str, step_key: str, page_data: Dict[str, Any]) -> bool:
        """
        Cache processed data from a page
        
        Args:
            workflow_id: Workflow ID
            step_key: Step key
            page_data: Page data containing filled fields and answers
            
        Returns:
            bool: Whether caching was successful
        """
        try:
            # Get or create step instance
            step = self.step_repo.get_step_by_key(workflow_id, step_key)
            if not step:
                logger.error(f"Step not found: {workflow_id}/{step_key}")
                return False
            
            # Prepare cache data
            cache_data = {
                "workflow_id": workflow_id,
                "step_key": step_key,
                "processed_at": datetime.utcnow().isoformat(),
                "page_data": page_data,
                "extracted_info": self._extract_key_info(page_data, step_key)
            }
            
            # Update step data
            current_data = step.data or {}
            current_data["cross_page_cache"] = cache_data
            
            self.step_repo.update_step_data(step.step_instance_id, current_data)
            
            logger.info(f"Cached data for {workflow_id}/{step_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache page data: {str(e)}")
            return False
    
    def get_previous_page_data(self, workflow_id: str, current_step_key: str, 
                              data_type: str = None) -> Dict[str, Any]:
        """
        Get relevant data from previous pages
        
        Args:
            workflow_id: Workflow ID
            current_step_key: Current step key
            data_type: Data type filter (address, personal, employment, etc.)
            
        Returns:
            Dict: Contains relevant data from previous pages
        """
        try:
            # Get all steps
            all_steps = self.step_repo.get_workflow_steps(workflow_id)
            
            # Get current step order
            current_step = self.step_repo.get_step_by_key(workflow_id, current_step_key)
            if not current_step:
                return {}
            
            current_order = current_step.order or 0
            
            # Collect data from previous steps
            previous_data = {}
            for step in all_steps:
                if (step.order or 0) < current_order and step.data:
                    cache_data = step.data.get("cross_page_cache")
                    if cache_data and self._is_cache_valid(cache_data):
                        # Filter by data type
                        if data_type:
                            filtered_info = self._filter_by_data_type(
                                cache_data.get("extracted_info", {}), 
                                data_type
                            )
                            if filtered_info:
                                previous_data[step.step_key] = filtered_info
                        else:
                            previous_data[step.step_key] = cache_data.get("extracted_info", {})
            
            return previous_data
            
        except Exception as e:
            logger.error(f"Failed to get previous page data: {str(e)}")
            return {}
    
    def analyze_address_completion(self, workflow_id: str, current_step_key: str, 
                                 profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze address completion status
        Specifically for handling "Do you have any other addresses" type questions
        
        Args:
            workflow_id: Workflow ID
            current_step_key: Current step key
            profile_data: User profile data
            
        Returns:
            Dict: Address completion analysis result
        """
        try:
            # Get previously filled address information
            previous_addresses = self.get_previous_page_data(
                workflow_id, current_step_key, "address"
            )
            
            # Extract all address information from profile_data
            all_addresses = self._extract_all_addresses_from_profile(profile_data)
            
            # Analyze filled and unfilled addresses
            analysis_result = {
                "filled_addresses": [],
                "remaining_addresses": [],
                "has_other_addresses": False,
                "completion_percentage": 0
            }
            
            # Collect filled addresses
            filled_address_keys = set()
            for step_key, step_data in previous_addresses.items():
                if "addresses" in step_data:
                    for addr in step_data["addresses"]:
                        filled_address_keys.add(addr.get("key", ""))
                        analysis_result["filled_addresses"].append({
                            "step_key": step_key,
                            "address": addr
                        })
            
            # Find unfilled addresses
            for addr_key, addr_data in all_addresses.items():
                if addr_key not in filled_address_keys:
                    analysis_result["remaining_addresses"].append({
                        "key": addr_key,
                        "data": addr_data
                    })
            
            # Determine if there are other addresses
            analysis_result["has_other_addresses"] = len(analysis_result["remaining_addresses"]) > 0
            
            # Calculate completion percentage
            total_addresses = len(all_addresses)
            if total_addresses > 0:
                analysis_result["completion_percentage"] = (
                    len(analysis_result["filled_addresses"]) / total_addresses * 100
                )
            
            logger.info(f"Address completion analysis for {workflow_id}/{current_step_key}: "
                       f"{len(analysis_result['filled_addresses'])} filled, "
                       f"{len(analysis_result['remaining_addresses'])} remaining")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Address completion analysis failed: {str(e)}")
            return {
                "filled_addresses": [],
                "remaining_addresses": [],
                "has_other_addresses": False,
                "completion_percentage": 0,
                "error": str(e)
            }
    
    def get_contextual_answers(self, workflow_id: str, current_step_key: str, 
                              questions: List[Dict[str, Any]], 
                              profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide contextual answers for current page questions based on previous page data
        
        Args:
            workflow_id: Workflow ID
            current_step_key: Current step key
            questions: List of questions on current page
            profile_data: User profile data
            
        Returns:
            Dict: Contextual answer suggestions
        """
        try:
            contextual_answers = {}
            
            # Get data from previous pages
            previous_data = self.get_previous_page_data(workflow_id, current_step_key)
            
            # Analyze each question
            for question in questions:
                question_text = question.get("question", "").lower()
                field_name = question.get("field_name", "")
                
                # Special handling: Address-related questions
                if any(keyword in question_text for keyword in 
                       ["other address", "additional address", "another address", "还有", "其他地址"]):
                    
                    address_analysis = self.analyze_address_completion(
                        workflow_id, current_step_key, profile_data
                    )
                    
                    contextual_answers[field_name] = {
                        "suggested_answer": "yes" if address_analysis["has_other_addresses"] else "no",
                        "confidence": 0.9,
                        "reasoning": f"Based on previous pages, {len(address_analysis['filled_addresses'])} "
                                   f"addresses have been filled, "
                                   f"{len(address_analysis['remaining_addresses'])} remain",
                        "context_data": address_analysis
                    }
                
                # Special handling: Parent details related questions
                elif any(keyword in question_text for keyword in 
                         ["parent details", "parents' details", "父母详情", "父母信息"]):
                    
                    parent_analysis = self._analyze_parent_details_completion(
                        previous_data, profile_data
                    )
                    
                    contextual_answers[field_name] = {
                        "suggested_answer": parent_analysis["suggested_answer"],
                        "confidence": parent_analysis["confidence"],
                        "reasoning": parent_analysis["reasoning"],
                        "context_data": parent_analysis
                    }
            
            return contextual_answers
            
        except Exception as e:
            logger.error(f"Contextual answers generation failed: {str(e)}")
            return {}
    
    def _extract_key_info(self, page_data: Dict[str, Any], step_key: str) -> Dict[str, Any]:
        """Extract key information from page data"""
        extracted = {
            "step_key": step_key,
            "processed_fields": []
        }
        
        # Extract form data
        form_data = page_data.get("form_data", [])
        if isinstance(form_data, list):
            for item in form_data:
                if isinstance(item, dict) and item.get("answer"):
                    field_info = {
                        "field_name": item.get("field_name", ""),
                        "question": item.get("question", ""),
                        "answer": item.get("answer", ""),
                        "field_type": item.get("field_type", "")
                    }
                    extracted["processed_fields"].append(field_info)
        
        # Special handling: Address information
        if "address" in step_key.lower() or "contact" in step_key.lower():
            extracted["addresses"] = self._extract_address_info(form_data)
        
        # Special handling: Family information
        if "family" in step_key.lower() or "parent" in step_key.lower():
            extracted["family_info"] = self._extract_family_info(form_data)
        
        return extracted
    
    def _extract_address_info(self, form_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract address information"""
        addresses = []
        
        for item in form_data:
            if isinstance(item, dict):
                field_name = item.get("field_name", "").lower()
                if any(keyword in field_name for keyword in 
                       ["address", "street", "city", "postcode", "country"]):
                    addresses.append({
                        "key": field_name,
                        "value": item.get("answer", ""),
                        "question": item.get("question", "")
                    })
        
        return addresses
    
    def _extract_family_info(self, form_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract family information"""
        family_info = {}
        
        for item in form_data:
            if isinstance(item, dict):
                field_name = item.get("field_name", "").lower()
                if any(keyword in field_name for keyword in 
                       ["parent", "father", "mother", "spouse", "family"]):
                    family_info[field_name] = {
                        "value": item.get("answer", ""),
                        "question": item.get("question", "")
                    }
        
        return family_info
    
    def _extract_all_addresses_from_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all address information from profile_data"""
        addresses = {}
        
        # Flatten profile_data
        flattened = self._flatten_dict(profile_data)
        
        # Find address-related fields
        for key, value in flattened.items():
            if value and isinstance(value, str) and len(value.strip()) > 0:
                key_lower = key.lower()
                if any(keyword in key_lower for keyword in 
                       ["address", "street", "city", "postcode", "country", "location"]):
                    addresses[key] = {
                        "value": value,
                        "type": self._classify_address_type(key_lower)
                    }
        
        return addresses
    
    def _classify_address_type(self, field_name: str) -> str:
        """Classify address type"""
        if "current" in field_name or "present" in field_name:
            return "current"
        elif "previous" in field_name or "former" in field_name:
            return "previous"
        elif "correspondence" in field_name or "postal" in field_name:
            return "correspondence"
        elif "work" in field_name or "office" in field_name:
            return "work"
        else:
            return "other"
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cache is valid"""
        try:
            processed_at = datetime.fromisoformat(cache_data.get("processed_at", ""))
            expiry_time = processed_at + timedelta(seconds=self.cache_ttl)
            return datetime.utcnow() < expiry_time
        except:
            return False
    
    def _filter_by_data_type(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Filter data by data type"""
        if data_type == "address":
            return {"addresses": data.get("addresses", [])}
        elif data_type == "family":
            return {"family_info": data.get("family_info", {})}
        else:
            return data
    
    def _analyze_parent_details_completion(self, previous_data: Dict[str, Any], 
                                         profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parent details completion status"""
        # Check if parent information was already handled in previous pages
        parent_info_filled = False
        parent_choice_made = False
        
        for step_key, step_data in previous_data.items():
            family_info = step_data.get("family_info", {})
            for field_name, field_data in family_info.items():
                if "parent" in field_name.lower():
                    if "no" in field_data.get("value", "").lower():
                        parent_choice_made = True
                    elif field_data.get("value", "").strip():
                        parent_info_filled = True
        
        # Provide suggestion based on analysis results
        if parent_choice_made:
            return {
                "suggested_answer": "no",
                "confidence": 0.9,
                "reasoning": "User previously indicated they don't have parent details"
            }
        elif parent_info_filled:
            return {
                "suggested_answer": "yes",
                "confidence": 0.8,
                "reasoning": "Parent information was already provided in previous page"
            }
        else:
            return {
                "suggested_answer": "unknown",
                "confidence": 0.5,
                "reasoning": "No clear indication from previous pages"
            }
    
    def cleanup_expired_cache(self, workflow_id: str) -> int:
        """Clean up expired cache data"""
        try:
            cleaned_count = 0
            all_steps = self.step_repo.get_workflow_steps(workflow_id)
            
            for step in all_steps:
                if step.data and "cross_page_cache" in step.data:
                    cache_data = step.data["cross_page_cache"]
                    if not self._is_cache_valid(cache_data):
                        # Remove expired cache
                        del step.data["cross_page_cache"]
                        self.step_repo.update_step_data(step.step_instance_id, step.data)
                        cleaned_count += 1
            
            logger.info(f"Cleaned {cleaned_count} expired cache entries for workflow {workflow_id}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
            return 0 