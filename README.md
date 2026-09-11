# Krishi Setu ML API

FastAPI service used by the Krishi Setu website for:

- Price prediction: `POST /predict-price`
- Demand prediction: `POST /predict-demand`
- Product recommendation: `POST /recommend-products`
- Health check: `GET /health`

## Render start command

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

## Krishi Setu request

The website sends requests directly to:

```text
https://sih-2026-lx6e.onrender.com
```

Example price/demand body:

```json
{
  "location": "Patia",
  "product": "Tomato",
  "market_supply_kg": 500,
  "festival_season": 0
}
```

## Important Render note

Render Free services can sleep when idle. The first request after sleeping can therefore take much longer than a normal request. This API exposes `/health` for Render health checks, but application-side code cannot remove the platform's cold-start delay.
