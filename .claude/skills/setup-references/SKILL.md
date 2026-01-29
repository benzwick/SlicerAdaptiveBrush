# Setup References Skill

Set up the `_reference/` folder with reference code for development.

## When to Use

- When a developer first clones the repository
- When adding new reference repositories
- When switching between symlink and clone

## Available References

| Name | Repository | Description |
|------|------------|-------------|
| SlicerSource | https://github.com/Slicer/Slicer | 3D Slicer source code |
| SlicerSegmentEditorExtraEffects | https://github.com/lassoan/SlicerSegmentEditorExtraEffects | Extra segment editor effects |
| QuantitativeReporting | https://github.com/QIICR/QuantitativeReporting | DICOM SEG with dcmqi |
| CrossSegmentationExplorer | https://github.com/ImagingDataCommons/SlicerCrossSegmentationExplorer | Segmentation comparison tool |

## Workflow

1. Ask which references to set up using AskUserQuestion
2. For each selected reference, ask: clone or symlink?
3. If symlink, ask for local path
4. Create the reference (clone or symlink)
5. Update .env if needed (e.g., SLICER_SOURCE path)
6. Verify setup

## Implementation

```python
# References configuration
REFERENCES = {
    "SlicerSource": {
        "url": "https://github.com/Slicer/Slicer.git",
        "description": "3D Slicer source code - large repo, symlink recommended",
        "default": "symlink",
        "env_var": "SLICER_SOURCE",
    },
    "SlicerSegmentEditorExtraEffects": {
        "url": "https://github.com/lassoan/SlicerSegmentEditorExtraEffects.git",
        "description": "Extra segment editor effects - good examples",
        "default": "clone",
    },
    "QuantitativeReporting": {
        "url": "https://github.com/QIICR/QuantitativeReporting.git",
        "description": "DICOM SEG handling with dcmqi",
        "default": "clone",
    },
    "CrossSegmentationExplorer": {
        "url": "https://github.com/ImagingDataCommons/SlicerCrossSegmentationExplorer.git",
        "description": "Segmentation comparison tool",
        "default": "clone",
    },
}
```

## Commands

### Clone a reference
```bash
cd _reference
git clone --depth 1 <url> <name>
```

### Symlink a reference
```bash
cd _reference
ln -s /path/to/local/repo <name>
```

### Verify setup
```bash
ls -la _reference/
```

## Example Session

User: /setup-references

Claude:
1. "Which references would you like to set up?"
   - [ ] SlicerSource (3D Slicer source code)
   - [ ] SlicerSegmentEditorExtraEffects (segment editor examples)
   - [ ] QuantitativeReporting (DICOM SEG)

2. For SlicerSource: "Clone or symlink?"
   - Symlink (recommended for large repos)
   - Clone

3. If symlink: "Enter path to your local Slicer source:"
   - User enters: /home/user/projects/Slicer

4. Execute setup and verify

## Notes

- SlicerSource is large (~2GB), symlink strongly recommended
- Shallow clones (--depth 1) save space for other repos
- The _reference folder is gitignored, so each dev has their own setup
