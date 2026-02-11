# Ash API Key Handling Overview

Ash uses a controlled, site-scoped mechanism for managing API keys in order to prevent accidental exposure, limit unauthorized use, and maintain operational control. The system is not designed to provide cryptographic secrecy—because Ash runs at user level, perfect secrecy is not possible—but it does provide meaningful protection against casual misuse and unintended disclosure.

1. API Key Storage Model
Ash never distributes the real API key directly to users. Instead, each site installation contains a site-wide secret key stored in .INSTALL, a single encrypted API key blob generated using that secret key, and a per-user token that authorizes access to the encrypted blob. The real API key is only decrypted in memory at runtime when Ash needs to make a request to an AI provider.

2. Purpose of the Encrypted Blob
The encrypted API key blob is not intended to provide strong cryptographic protection. Its purpose is operational: preventing the real API key from appearing in plaintext in environment variables, shell history, logs, or configuration files; preventing users from accidentally pasting the real key into external tools; preventing casual misuse by making it non-trivial to extract the real key for use in other applications; allowing site administrators to revoke access by rotating the site secret key; allowing per-user access control through token issuance and deny-lists.

3. Threat Model and Security Intent
Ash’s API key handling is designed to protect against accidental leakage of the API key, users copying the key into other tools, unauthorized scripts or applications using the key, key sprawl across machines or user accounts, and inadvertent disclosure in logs or shell history. It does not protect against a malicious user intentionally reverse-engineering Ash, modifying Ash to print the decrypted key, or extracting secrets with full system access. The system is intentionally simple, dependency-free, and auditable.

4. Access Control and Revocation
Ash supports site-wide revocation by rotating the site secret key, user-level revocation through deny-lists, and token-based authorization where each user receives a token that must match the site secret key to decrypt the API key blob. These mechanisms ensure that only authorized users can run Ash with the site’s API key.

5. Rationale for the Design
Because Ash runs at user level, no mechanism can fully prevent a determined user from extracting the API key. The encrypted blob approach minimizes risk of accidental exposure, prevents casual misuse, keeps the real API key out of user-visible locations, provides revocation and access control, maintains portability and simplicity, and avoids reliance on OS-level secret stores or privileged access. This design strikes a balance between practicality, safety, and operational control.
