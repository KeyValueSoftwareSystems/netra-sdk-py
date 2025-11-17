# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [0.1.53] - 2025-11-17

- Added custom instrumentation for ADK framework
- Refactored DSPy instrumentation

## [0.1.52] - 2025-11-11

- Fixed attribute max length issue

## [0.1.51] - 2025-11-10

- Added custom instrumentation for Cerebras framework
- Fixed bug in traceloop instrumentation

## [0.1.50] - 2025-11-07

- Added custom dataset and entries

## [0.1.49] - 2025-11-06

- Fixed token count calculation for OpenAI response API

## [0.1.48] - 2025-11-05

- Added custom instrumentation for Groq framework

## [0.1.47] - 2025-10-21

- Added support for existing tracer provider usage

## [0.1.46] - 2025-10-17

- Fixed exception during add conversation
- Added support for observation type in spans

## [0.1.45] - 2025-09-29

- Added utility to locally block specific spans within a particular span scope.

## [0.1.44] - 2025-09-29

- Added utility to globally block specific spans from being exported to the tracing backend.

## [0.1.43] - 2025-09-17

- Fixed conversation content length issue
- Added utils module to handle common tasks

## [0.1.42] - 2025-09-09

- Refactored conversation attribute format to be more consistent with OpenTelemetry

## [0.1.41] - 2025-09-09

- Refactored codebase to remove duplicate code

## [0.1.40] - 2025-09-08

- Added span level conversation support

## [0.1.39] - 2025-09-02

- Refactored code to remove duplicate code

## [0.1.38] - 2025-09-02

- Fixed instrumentation name detection issue

## [0.1.37] - 2025-09-01

- Fixed context detachment issue in session manager

## [0.1.36] - 2025-09-01

- Added a trace level method set_prompt to set prompt on any active span

## [0.1.35] - 2025-09-01

- Patch fix for set_input and set_output methods to set attributes on root span if no span is provided
- Patch fix to create streaming aware decorators

## [0.1.34] - 2025-08-29

- Changed block spans from being exported to block root level spans from being exported

## [0.1.33] - 2025-08-29

- Added utility to block specific spans from being exported to the tracing backend.
- Fixed context detachment issue in span wrapper.

## [0.1.32] - 2025-08-28

- Added support for scrubbing sensitive data from spans.

## [0.1.31] - 2025-08-28

- Added custom instrumentation for LiteLLM framework

## [0.1.30] - 2025-08-27

- Added utility to set input and output data for any active span in a trace

[0.1.53]: https://github.com/KeyValueSoftwareSystems/netra-sdk-py/tree/main
