# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 1.x (main) | Yes |
| < 1.0 | No |

## Reporting a vulnerability

**Do not open a public issue for security reports.** Instead email
sanskarpandey2004@gmail.com with:

- A description of the issue and its impact
- Steps to reproduce (minimal script preferred)
- Affected version / commit
- Any suggested mitigation

You will receive an acknowledgement within 72 hours. We will coordinate a fix
and disclosure timeline with you, and credit you unless you prefer anonymity.

## Scope

This is a statistics/visualization library and demo web app. The most
security-relevant surfaces are dependency vulnerabilities (`pip-audit` runs in
CI), the Docker image (non-root `appuser`, pinned slim base), and the Streamlit
server configuration. Reports about those areas are welcome, as are reports of
accidental secret commits (we will purge and rotate).
