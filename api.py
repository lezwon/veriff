import time
from typing import List, Optional, Dict

import uvicorn
from classifier import BirdClassifier, model_url, labels_url
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

app = FastAPI()
classifier = BirdClassifier(model_url, labels_url)


class ImageInput(BaseModel):
    """ " Input class for image urls"""

    images: List[HttpUrl] = None
    k: Optional[int] = 3


@app.post("/")
async def index(images: ImageInput) -> Dict:
    """Takes image url as input and returns top 3 results as output"""
    start_time = time.time()

    successful_results, failed_results, errors = [], [], []

    if not images.images:
        errors.append("No images provided")
    elif len(images.images) > 5:
        errors.append("Too many images provided. Max 5 images allowed")
    else:
        successful_results, failed_results = await classifier.run(
            images.images, images.k
        )

    return {
        "successful_results": successful_results,
        "failed_results": failed_results,
        "time_taken": time.time() - start_time,
        "errors": errors,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
