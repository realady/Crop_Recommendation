#!/bin/bash

# Start the FastAPI application with uvicorn
uvicorn main:app --host=0.0.0.0 --port=$PORT
