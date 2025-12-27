## Overview

The Video Inference Service is an independent, open-source software component for
AI-based image and video analysis.

It provides object detection and pose analysis via a simple HTTP API and is
designed to run as a standalone service in on-premise environments.

The service can be integrated with external systems via standard network
interfaces but does not include any user interface or business logic.

## Key Characteristics

- Standalone inference service (no UI, no orchestration logic)
- HTTP/REST API-based integration
- Designed for on-premise operation
- Replaceable and technology-agnostic interface
- No dependency on specific client applications

## Architectural Separation

This repository contains only the Video Inference Service.

It is intentionally designed as a technically and legally independent component.
Any consuming system interacts with this service exclusively via a documented
network API.

This service is not embedded, linked, or integrated into any other software
product.

## License

This software is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)**.

You may use, modify, and redistribute this software in accordance with the terms
of the AGPL-3.0.

License text:
https://www.gnu.org/licenses/agpl-3.0.html

## Notice on Modifications and Network Use

If you modify this software and make it available for use over a network,
you are required to provide the complete corresponding source code of your
modified version in accordance with the AGPL-3.0.

This obligation applies independently of whether the software is operated
locally or remotely.

## Third-Party Software

This project uses third-party open-source components.

A complete list of included software and their respective licenses can be found
in the file:

- THIRD_PARTY_NOTICES.md

## Installation

### Requirements
- Python 3.x
- Docker (optional, recommended)

### Local Installation

pip install -r requirements.txt
python app.py

## API

The service exposes a simple HTTP API for inference requests.

Example endpoints:
- GET /health
- POST /analyze

See the source code for request and response formats.

## No Warranty

This software is provided "as is", without warranty of any kind, express or
implied, except where required by applicable law.

There is no obligation to provide support, maintenance, or updates unless
explicitly agreed otherwise in writing.

## Maintainer

DiscoverBI UG (haftungsbeschränkt)

Website: https://www.discover-bi.com  
Contact: info@discover-bi.com
