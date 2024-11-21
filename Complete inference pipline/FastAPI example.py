from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body model
class CalculationRequest(BaseModel):
    number: float

# Endpoint to handle requests
@app.post("/calculate")
def calculate(request: CalculationRequest):
    try:
        # Perform calculations (e.g., square the input number)
        number = request.number
        result = number ** 2  # Example calculation: square the number

        # Return the result as a JSON response
        return {"number": number, 
                "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str('test error'))

# Run the server (using Uvicorn in the terminal)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
