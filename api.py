import logging
import time
from typing import List, Optional, Dict, cast

import constants
import uvicorn
from classifier import BirdClassifier
from constants import model_url, labels_url
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()
classifier = BirdClassifier(model_url, labels_url)

logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    """Input class for image urls"""

    images: List[HttpUrl] = []
    k: Optional[int] = 3


@app.post("/")
async def index(images: ImageInput) -> Dict:
    """Takes image url as input and returns top 3 results as output"""
    logger.info("Request received")
    logger.info(f"Images: {images.images}")
    logger.info(f"K: {images.k}")

    start_time = time.time()

    successful_results: Dict[str, List] = {}
    failed_results: List = []
    errors: List = []

    if not images.images:
        errors.append("No images provided")
    elif len(images.images) > constants.max_images:
        errors.append(
            f"Too many images provided. Max {constants.max_images} images allowed"
        )
    elif images.k > constants.max_k:
        errors.append(f"Max allowed value of K is {constants.max_k}")
    elif images.k < constants.min_k:
        errors.append(f"Min allowed value of K is {constants.min_k}")
    else:
        successful_results, failed_results = await classifier.run(
            cast(List[str], images.images), images.k
        )

    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "time_taken": time.time() - start_time,
        "errors": errors,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
