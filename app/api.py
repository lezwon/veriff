import time
from typing import List, Optional, Dict, cast

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from app import constants
from app.bird_classifier import BirdClassifier
from app.logger import Logger

server = FastAPI()
classifier = BirdClassifier(constants.model_url, constants.labels_url)


logger = Logger.getLogger(__name__, True)


class ImageInput(BaseModel):
    """Input class for image urls"""

    images: List[HttpUrl] = []
    k: Optional[int] = 3


@server.post("/")
async def index(input_: ImageInput) -> Dict:
    """Takes image url as input and returns top 3 results as output"""
    logger.info("Request received")
    logger.info(f"Number of Images: {len(input_.images)}")
    logger.info(f"K: {input_.k}")

    start_time = time.time()

    successful_results: Dict[str, List] = {}
    failed_results: List = []
    errors: List = []

    if not input_.images:
        errors.append("No images provided")
    elif len(input_.images) > constants.max_images:
        errors.append(
            f"Too many images provided. Max {constants.max_images} images allowed"
        )
    elif input_.k > constants.max_k:
        errors.append(f"Max allowed value of K is {constants.max_k}")
    elif input_.k < constants.min_k:
        errors.append(f"Min allowed value of K is {constants.min_k}")
    else:
        successful_results, failed_results = await classifier.run(
            cast(List[str], input_.images), input_.k
        )

    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "time_taken": time.time() - start_time,
        "errors": errors,
    }


if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8000)
