import uvicorn
from app.api import server

if __name__ == "__main__":
    uvicorn.run(server)
