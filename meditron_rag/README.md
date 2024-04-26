docker images
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
Access web UI at http://localhost:6333/dashboard

To run use following command:
uvicorn rag:app