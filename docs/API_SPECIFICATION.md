# AutoRoll API Specification

## REST API (v1)

### Health Check
`GET /api/v1/health`

Response:
```json
{
  "status": "healthy",
  "service": "AutoRoll Central Server",
  "version": "0.1.0"
}
```

## WebSockets

- `/ws/workers?worker_id={id}`: Worker registration and control stream.
- `/ws/clients`: Frontend client live attendance update stream.
