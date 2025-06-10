import asyncio
import json
import threading
import uuid
from typing import Optional, List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.document_classifier import DocumentClassifier
from src.document_struction import DocumentResolver
from src import graph

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6011,
        reload=True
    )

class MessageRequest(BaseModel):
    message: str
    trace_id: Optional[str] = None
    fill_data: Optional[dict] = None


class FileRequest(BaseModel):
    doc_urls: List[str]
    format_json: Optional[str] = None

thread_local = threading.local()


@app.post("/assistant")
async def run_workflow(message: MessageRequest):
    # 直接调用langgraph应用
    if message.trace_id is None:
        message.trace_id = str(uuid.uuid4())

    graph.global_static_map.update({message.trace_id: message.fill_data})

    msg = graph.system_message.format_messages(messages=message.message)
    result = graph.run_with_sidecar(
        {"messages": msg},
        config={"configurable": {"thread_id": message.trace_id}}
    )
    last_message = result["messages"][-1]
    content = last_message.content
    if not content:
        return {"error": "Content is empty"}

    try:
        json_data = json.loads(content) if isinstance(content, str) else content

    except json.JSONDecodeError:
        return {"error": "Invalid JSON in content"}

    graph.global_static_map.pop(message.trace_id)
    return {
        "result": json_data
    }

@app.post("/files/classify")
async def analyze_image_classify_endpoint( # Renamed to avoid conflict
        file_request: FileRequest, # Keep this as single URL for now, or update if needed
):
    try:

        if not file_request.doc_urls:
             return JSONResponse(status_code=400, content={"message": "No document URLs provided"})
        doc_url_to_classify = file_request.doc_urls[0]

        # Consider making API key an environment variable
        classifier = await DocumentClassifier(doc_url_to_classify).classify_document() # Use the selected URL

        return classifier

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error performing classification : {str(e)}"}
        )


@app.post("/files/structure")
async def files_structure_endpoint(
    file_request: FileRequest, # Use the updated FileRequest with doc_urls
):
    tasks = []
    # Create a task for each URL
    for doc_url in file_request.doc_urls:
        tasks.append(DocumentResolver(doc_url).resolver()) # Create resolver tasks

    try:
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_results = {}
        # Combine results from all successful tasks, filling fields from the first non-null value
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Log or handle error for specific URL
                print(f"Error processing URL {file_request.doc_urls[i]}: {result}")
                # Optionally add error info to the response, e.g.:
                # combined_results[f"error_url_{i}"] = f"Failed to process {file_request.doc_urls[i]}: {result}"
            elif isinstance(result, dict): # Check if the result is a dictionary
                for field, value in result.items():
                    if field not in combined_results and value is not None:
                        combined_results[field] = value


        if not combined_results:
             return JSONResponse(status_code=404, content={"message": "No non-null fields could be extracted or all tasks failed."})

        return combined_results # Return the combined dictionary

    except Exception as e:
        # This catches errors during asyncio.gather or top-level processing
        return JSONResponse(
            status_code=500,
            content={
                "message": f"Error processing documents: {str(e)}"}\
        )
