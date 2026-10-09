# Glama's introspection image for `laya-mcp`.
#
# Glama builds this image in its own sandbox and speaks MCP to it over stdio to
# capture `initialize` and `tools/list`. Those two requests are the whole job,
# and the image is built so that they are all it needs:
#
#   * `tools/list` does not need the model. `laya_mcp.mcp_server` registers the
#     five tools statically and only reaches for a checkpoint when a tool is
#     *called* - `Backend.start_warm` loads on a background thread precisely so
#     the handshake is never made to wait for weights. So the handshake is
#     answered with no checkpoint, no cache, no credential and no network.
#
#   * `laya` is therefore installed **not at all**, even though `pyproject.toml`
#     names it a hard dependency. Installing it drags in torch + transformers
#     (multiple GB) and a ~650 MB checkpoint fetch on the first run, and changes
#     nothing about the two requests this image exists to answer. `--no-deps` is
#     what skips it, and the MCP SDK - the project's own `mcp` extra, declared as
#     `mcp>=1.28,<2` - is installed explicitly instead. It is pinned to one
#     version because that SDK's v2 release removed the `FastMCP` export this
#     server is built on.
#
#   * A tool *call* with no model behind it still returns the package's own
#     structured refusal (`{"ok": false, "error": ...}`) instead of hanging or
#     crashing. This image cannot answer questions about a document; it exists so
#     the tool definitions can be read and graded.
#
# The base image is pinned to a Debian release rather than a digest on purpose:
# a digest could not be verified in the environment this file was written in
# (no Docker daemon), and an unverifiable single point of failure in a build that
# cannot be test-built is worse than a tag that still receives security patches.
# Everything that decides which code runs - Python 3.12, the SDK version, the
# package version - is pinned exactly.

FROM python:3.12-slim-bookworm

# PYTHONUNBUFFERED matters more than usual here: this process is a stdio server,
# and a reply that sits in a pipe buffer is a reply the client never reads.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Before the source copy, so editing the package does not refetch any of these.
#
# setuptools and wheel are for the build below, not for the runtime, and they are
# pinned for the same reason the SDK is: pip's isolated build environment would
# otherwise resolve `setuptools>=68` to whatever PyPI serves on the day of the
# build, and the build backend is what decides the contents of the wheel. Pinned
# so the image stays a function of this file.
RUN python -m pip install --no-cache-dir \
      "mcp==1.30.0" \
      "setuptools==80.9.0" \
      "wheel==0.45.1"

COPY . /app

# `--no-deps` is the substantive decision here: it installs laya-mcp without
# `laya`. `--no-build-isolation` pairs with the pinned setuptools above.
#
# `README.md` and `LICENSE` have to survive `.dockerignore` for this step to work
# at all - `pyproject.toml` reads the former (`readme =`) and ships the latter
# (`license-files =`) - which is why that file excludes `*.md` *except*
# README.md rather than every markdown file. `src/laya_mcp/SKILL.md` is a third
# such file for the same reason: `[tool.setuptools.package-data]` ships it inside
# the package, so the ignore file re-includes it too. All three were checked
# against exactly the file set `.dockerignore` produces: the build emits
# `laya_mcp-0.2.3-py3-none-any.whl` with `laya_mcp/SKILL.md` inside it, and
# installs with `laya` and `torch` both absent from the environment.
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps .

# The `mcp` subcommand is not decoration. `python -m laya_mcp` with no subcommand
# is documented as "be an MCP server", but that branch builds a Namespace holding
# only `sidecar` and `filter` and then reads `args.model`, so it raises
# AttributeError and exits 1 before answering anything. Measured on this source
# tree: bare `python -m laya_mcp` exits 1 with `AttributeError: 'Namespace'
# object has no attribute 'model'`, while `python -m laya_mcp mcp` completes the
# handshake and lists all five tools.
ENTRYPOINT ["python", "-m", "laya_mcp", "mcp"]
