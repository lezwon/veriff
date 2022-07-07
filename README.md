# Veriff Assignment

## How to Setup

1. Create venv or conda environment with python3.7 `conda create -n veriff`
2. Activate environment and install requirements. `pip install -r requirements.txt`
3. Run `uvicorn main:app --port 8000`
4. The server will be started on port `8000`.

## Dev / Tests Setup
1. Follow the steps above to setup the environment
2. Install dev requirements with `pip install -r requirements.dev.txt`
3. Setup pre-commit hooks with `pre-commit install`
4. To run tests, call `pytest .`

## Docker Setup
1. Build the docker image
    - CPU: `docker build -t veriff .`
    - GPU: `docker build -t veriff -f Dockerfile.gpu .`
2. Run the docker image
    - CPU: `docker run -d -p 8000:8000 --name server  veriff`
    - GPU: `docker run -d -p 8000:8000 --name server --gpus all veriff`
3. Server will be available on port `8000`.

## Inference API
1. You can call the inference API at `/` with a `POST` request.
2. The request body is `json`
3. The response body is `json` and contains the fields:
    - `successful_results`: `Dict[str, List]` Dictionary of successful predictions
    - `failed_results`: `List` Urls of images which failed to be processed
    - `time_taken`: `float` Time taken to process the request
    - `errors`: `List` List of all validation errors

```shell
curl --location --request POST 'http://127.0.0.1:8000' \
--header 'Content-Type: application/json' \
--data-raw '{
    "images":[
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg",
        "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg"
    ],
    "k": 4
}'
```

Sample Response:
```json
{
    "successful_results": {
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Phalacrocorax_varius_-Waikawa%2C_Marlborough%2C_New_Zealand-8.jpg": [
            {
                "bird_name": "Phalacrocorax varius varius",
                "score": 0.8430764079093933
            },
            {
                "bird_name": "Phalacrocorax varius",
                "score": 0.11654692888259888
            }
        ],
        "https://quiz.natureid.no/bird/db_media/eBook/679edc606d9a363f775dabf0497d31de8c3d7060.jpg": [
            {
                "bird_name": "Galerida cristata",
                "score": 0.8428874611854553
            },
            {
                "bird_name": "Alauda arvensis",
                "score": 0.08378683775663376
            }
        ]
    },
    "failed_results": [],
    "time_taken": 1.758669137954712,
    "errors": []
}
```

## Benchmarks

#### System hardware
- CPU: Intel(R) Core(TM) i5-7360U CPU @ 2.30GHz
- Cores: 2
- GPU: None
- Memory: 8GB

#### Configuration
| Setting              | Value |
|----------------------|-------|
| Total Calls          | 1000  |
| Concurrent Calls     | 16    |
| Timeout              | 10s   |
| Images               | 5     |
| Process workers      | 4     |
| Concurrent downloads | 10    |


#### Summary

| Type                | Time
|---------------------|--------------|
| Mean query speed    | 3642 ms      |
| Fastest query speed | 1990 ms      |
| Slowest query speed | 9460 ms      |
| Mean RPS            | 4.32 req/sec |
| Total time          | 230856 ms    |

Response Codes
200: 998 responses (100.00%)


### Current Bottlenecks and Future Improvements:
1. The current model inference is sequential. I would move the model inference to TFServing to utilize dynamic batching.
2. Using TFServing would also allow me to version the model.
3. I would use a load balancer to handle multiple requests.
4. Move error messages to a yaml file for internationalization.
5. Build docker image for GPU support.
6. Benchmark and optimize performance using NSights by Nvidia.
7. Handling large number of images by breaking them down into small batches
8. Use Kubernetes for deploying at scale (No experience working with it yet)
