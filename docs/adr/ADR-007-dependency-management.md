# ADR-007: Optional Dependency Management

## Status

Accepted

## Context

Some advanced features require optional Python packages not bundled with 3D Slicer:

- **scikit-learn**: GMM-based intensity analysis
- **scikit-image**: Random Walker algorithm

These packages enhance functionality but are not essential. Users should be able to:
1. Use the extension without optional dependencies (with fallback behavior)
2. Install dependencies when they first need them
3. Not be interrupted by installation prompts during normal workflow

Slicer's Python environment uses pip, but installation requires user consent and may fail on some systems.

## Decision

Implement a **DependencyManager** class that:

1. **Checks availability** without importing (avoids import errors)
2. **Prompts user** only when a feature requiring the dependency is selected
3. **Installs via slicer.util.pip_install()** with user confirmation dialog
4. **Falls back gracefully** when user declines or installation fails

### User Flow

1. User selects an algorithm requiring an optional dependency (e.g., Random Walker)
2. If dependency not available, show confirmation dialog:
   ```
   "Random Walker requires scikit-image. Install now? (requires internet)"
   [Install] [Cancel]
   ```
3. If user clicks Install:
   - Show progress indicator
   - Install via pip
   - Re-enable the feature
4. If user cancels:
   - Revert to previous algorithm selection
   - Show informational message

### Implementation

```python
class DependencyManager:
    """Manages optional Python dependencies."""

    _registry = {
        "sklearn": DependencySpec(
            import_name="sklearn",
            pip_name="scikit-learn",
            description="GMM intensity analysis",
        ),
        "skimage": DependencySpec(
            import_name="skimage",
            pip_name="scikit-image",
            description="Random Walker algorithm",
        ),
    }

    def is_available(self, name: str) -> bool:
        """Check if package is importable without importing."""

    def ensure_available(self, name: str) -> bool:
        """Prompt user to install if needed. Returns True if available."""
```

## Consequences

### Positive

- Extension works out-of-box with core features
- Users only install what they need
- Clear communication about what's being installed
- No silent failures or confusing error messages
- Respects user's choice not to install

### Negative

- First use of advanced features requires internet connection
- Installation may fail on restricted systems
- Adds complexity to algorithm selection logic

## Alternatives Considered

### Always require optional dependencies

**Rejected**: Would prevent extension from working on systems where pip install fails.

### Install on extension load

**Rejected**: Would slow startup and prompt users who may never need the features.

### Silent fallback without prompt

**Rejected**: Users wouldn't know why certain features behave differently.

## References

- [Slicer Python Package Installation](https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#install-a-python-package)
- `slicer.util.pip_install()` API
