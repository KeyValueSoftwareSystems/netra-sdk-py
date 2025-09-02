# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

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

[0.1.39]: https://github.com/KeyValueSoftwareSystems/netra-sdk-py/tree/main
