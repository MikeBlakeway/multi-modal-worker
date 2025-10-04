# Multi-Modal Inference Worker API Reference

## Overview

The Multi-Modal Inference Worker provides a unified API for multiple AI inference modalities through a single RunPod serverless endpoint. This document describes the API interface, request/response schemas, and usage examples.

## Supported Modalities

### Text-to-Image

- **Endpoint**: `POST /inference`
- **Modality**: `text-to-image`
- **Model**: FLUX.1 Schnell (fp8 quantized)
- **Description**: Generate high-quality images from text descriptions

### Image-to-Video

- **Endpoint**: `POST /inference`
- **Modality**: `image-to-video`
- **Model**: AnimateDiff motion adapter
- **Description**: Animate static images into video sequences

### Text-to-Video

- **Endpoint**: `POST /inference`
- **Modality**: `text-to-video`
- **Model**: LTX-Video 2B (distilled)
- **Description**: Generate video content from text descriptions

### ControlNet

- **Endpoint**: `POST /inference`
- **Modality**: `control-net`
- **Models**: Canny edge detection, Depth estimation
- **Description**: Generate images with precise structural control

### Inpainting

- **Endpoint**: `POST /inference`
- **Modality**: `inpainting`
- **Model**: SDXL Inpainting
- **Description**: Fill or modify specific regions of images

### Camera Control

- **Endpoint**: `POST /inference`
- **Modality**: `camera-control`
- **Model**: CameraCtrl
- **Description**: Apply camera movements and effects to video content

## Request Schema

```json
{
  "input": {
    "modality": "text-to-image | image-to-video | text-to-video | control-net | inpainting | camera-control",
    "prompt": "string (required for text-based modalities)",
    "image_url": "string (required for image-based modalities)",
    "mask_url": "string (required for inpainting)",
    "control_image_url": "string (required for ControlNet)",
    "steps": "integer (1-50, default: 4)",
    "guidance_scale": "float (0.0-20.0, default: 1.0)",
    "width": "integer (64-2048, default: 1024)",
    "height": "integer (64-2048, default: 1024)",
    "seed": "integer (optional, for reproducibility)",
    "num_frames": "integer (8-32, for video modalities)",
    "fps": "integer (8-30, for video modalities)"
  }
}
```

## Response Schema

```json
{
  "output": {
    "modality": "string",
    "result_type": "image | video",
    "result_url": "string",
    "metadata": {
      "inference_time": "float (seconds)",
      "model_used": "string",
      "parameters": "object"
    }
  }
}
```

## Usage Examples

### Text-to-Image Example

```json
{
  "input": {
    "modality": "text-to-image",
    "prompt": "A serene mountain landscape at sunset",
    "steps": 4,
    "guidance_scale": 1.0,
    "width": 1024,
    "height": 1024
  }
}
```

### Image-to-Video Example

```json
{
  "input": {
    "modality": "image-to-video",
    "image_url": "https://example.com/image.jpg",
    "num_frames": 16,
    "fps": 24
  }
}
```

## Error Handling

### Error Response Schema

```json
{
  "error": "string (error description)",
  "supported_modalities": ["array of supported modality strings"],
  "request_id": "string (for debugging)"
}
```

### Common Error Cases

- **400 Bad Request**: Invalid modality or missing required parameters
- **413 Payload Too Large**: Input images exceed size limits
- **422 Unprocessable Entity**: Invalid parameter values
- **500 Internal Server Error**: Model loading or inference errors

## Performance Targets

| Modality       | Target Time | Input Size      | Output Format |
| -------------- | ----------- | --------------- | ------------- |
| Text-to-Image  | <15s        | Text prompt     | PNG/JPEG      |
| Image-to-Video | <25s        | 1024x1024       | MP4           |
| Text-to-Video  | <45s        | Text prompt     | MP4           |
| ControlNet     | <20s        | Image + control | PNG/JPEG      |
| Inpainting     | <18s        | Image + mask    | PNG/JPEG      |
| Camera Control | <30s        | Video input     | MP4           |

## Model Information

### Storage Requirements

- Total model storage: ~40GB (within 80GB limit)
- Shared components optimized for memory efficiency
- Dynamic model loading to manage GPU memory

### Optimization Features

- fp8 quantization for reduced memory usage
- Smart model eviction based on usage patterns
- FlashBoot optimization for faster cold starts
- CPU offloading for memory management

## Integration Notes

This API is designed to integrate seamlessly with the existing Media Labs frontend architecture using the established hooks-based pattern and workflow template system.

---

_This is a placeholder document that will be updated as implementation progresses through the defined phases._
