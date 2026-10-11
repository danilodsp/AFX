# Security policy

## Supported versions

Security fixes go into the latest release on PyPI. Older versions are not
patched; upgrade with `pip install -U afx-audio`.

## Reporting a vulnerability

Please do not open a public issue. Email danilopena.ba@gmail.com with a
description of the problem, the affected version and steps to reproduce, or
use GitHub's private vulnerability reporting on the repository's Security tab.
You will get a reply within a week. Once a fix is released, the report is
credited in the changelog unless you ask otherwise.

AFX reads audio files through soundfile (libsndfile). Vulnerabilities in those
libraries should also be reported upstream.
