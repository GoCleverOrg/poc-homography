# Output Videos

This folder contains all processed video outputs from the homography stream annotation system.

## File Naming Convention

```
{CameraName}_output_stream_annotated_{YYYYMMDD_HHMMSS}.mp4
{CameraName}_output_stream_annotated_{YYYYMMDD_HHMMSS}_expanded.mp4
```

**Example:**
- `Valte_output_stream_annotated_20251121_182649.mp4` - Normal annotated video
- `Valte_output_stream_annotated_20251121_182649_expanded.mp4` - With side panel (top-down view)

## File Types

### Normal Annotated Video (`*.mp4`)
- Original camera resolution
- Bounding boxes with labels
- Foot points marked (used for homography)
- **Use for:** Reviewing detections in context

### Expanded Video (`*_expanded.mp4`)
- Original video + side panel
- Side panel shows top-down view with projected positions
- Camera position marked at bottom center
- **Use for:** Verifying homography accuracy and object positioning

## Camera Names

- **Valte**: Camera at 10.207.99.178
- **Setram**: Camera at 10.237.100.15

## Storage Management

Videos are automatically saved here by `main.py`. To clean up old videos:

```bash
# Delete videos older than 7 days
find output/ -name "*.mp4" -mtime +7 -delete

# Delete all videos (keep README)
find output/ -name "*.mp4" -delete
```

## Typical File Sizes

- 10 second capture: ~10-15 MB per file
- Normal video: ~1 MB/second
- Expanded video: ~1.2-1.5 MB/second (larger due to side panel)

---

**Note:** This folder is ignored by git (.gitignore). Videos are not committed to version control.
