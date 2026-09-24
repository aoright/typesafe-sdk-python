# TypeSafe AI Python SDK

Python SDK for [TypeSafe AI](https://typesafe.ai).

## Quickstart

Install the SDK:

```
uv add typesafe-sdk
```

Set `TYPESAFE_API_KEY` in your environment, then instantiate and use the client:

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "category": Choice(
                instructions="What is this ticket about?",
                criteria={"billing": None, "technical": None, "other": None},
            ),
        },
    )

print(response.choices["category"].choice)
```

learn what TypeSafe is, what it can do, and how to use it in [TypeSafe docs](https://docs.typesafe.ai/).

## HTTP/2 Support

Enabling HTTP/2 allows multiplexing multiple concurrent requests over a single TCP connection, significantly improving throughput and reducing latency when evaluating batches of requests or running concurrent agents.

1. Install `httpx2` with HTTP/2 dependencies:
   ```bash
   uv add "httpx2[http2]"
   ```

2. Pass an HTTP/2-enabled client to `TypeSafeClient` or `AsyncTypeSafeClient`:

   ```python
   import asyncio
   import httpx2
   from typesafe_sdk import AsyncTypeSafeClient, Noul

   async def main() -> None:
       # Enable HTTP/2 multiplexing via httpx2
       async with AsyncTypeSafeClient(
           http_client=httpx2.AsyncClient(http2=True)
       ) as client:
           tasks = [
               client.system_one(
                   state={"id": i, "content": f"Message {i}"},
                   questions={"urgent": Noul("Does this require immediate action?")},
               )
               for i in range(10)
           ]
           results = await asyncio.gather(*tasks)
           print(f"Evaluated {len(results)} concurrent requests over HTTP/2")

   asyncio.run(main())
   ```

Note: `TypeSafeClient.close()` and context manager exits close the underlying HTTP client. When sharing a single `AsyncTypeSafeClient` across tasks, manage its lifecycle at the application boundary rather than recreating it per request.

For further configuration options, refer to the [HTTPX2 HTTP/2 Documentation](https://pydantic.dev/docs/httpx2/httpcore2/http2/#enabling-http2).

## Documentation

Learn more in [SDK docs](https://docs.typesafe.ai/sdk/python/).
