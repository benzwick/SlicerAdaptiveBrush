# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the SlicerAdaptiveBrush project.

## What is an ADR?

An Architecture Decision Record captures an important architectural decision made along with its context and consequences.

## ADR Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](ADR-001-algorithm-selection.md) | Multi-Stage Hybrid Algorithm Selection | Accepted |
| [ADR-002](ADR-002-python-cpp-boundaries.md) | Python vs C++ Implementation Boundaries | Accepted |
| [ADR-003](ADR-003-testing-strategy.md) | Testing Strategy | Accepted |
| [ADR-004](ADR-004-caching-strategy.md) | Caching Strategy for Drag Operations | Accepted |

## Template

When adding a new ADR, use this template:

```markdown
# ADR-XXX: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?

## Alternatives Considered
What other options were evaluated?
```
