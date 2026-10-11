# Image placement

Create these folders and copy the corresponding images into them:

```text
docs/assets/
├── architecture.png                       ← architecture.png
├── pose-extraction/
│   ├── dance-scene.png                    ← dance_scene.png
│   ├── cropped-frame.png                  ← Cropped_Frame.png
│   ├── alphapose-overlay.png              ← AlpaPose_result.png
│   └── isolated-skeleton.png              ← Skeleton.png
└── skeleton-motion/
    ├── frame-104.png                      ← skeleton_frame_000104.png
    ├── frame-106.png                      ← skeleton_frame_000106.png
    ├── frame-109.png                      ← skeleton_frame_000109.png
    └── frame-110.png                      ← skeleton_frame_000110.png
```

The website expects precisely these lowercase target filenames.

The package already contains `architecture.png`, `dance-scene.png`,
`cropped-frame.png`, and `isolated-skeleton.png`. Add the AlphaPose overlay and
the four skeleton-motion frames from your original files before publishing.
