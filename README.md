# SOARcuit Shared

`soarcuit-shared` contains the small, durable core that multiple SOARcuit
services can share without inheriting Thalamus-specific policy.

Current scope:

- canonical domain models for inbound observations and memes
- stable enums and validation helpers
- Pub/Sub payload decoding, normalization, and message-kind detection

Out of scope:

- routing policy
- telemetry setup
- persistence implementations and schema-specific helpers

## Deployment

Publish to provate artifactory at  <https://us-east4-python.pkg.dev/soarcuit/soarcuit-dev-python-repo/simple/>
