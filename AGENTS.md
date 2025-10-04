# AGENTS.md – MVP Worker Requirements

## **MANDATORY AI ASSISTANT PROCESS CHECKLISTS FOR STORY WORK**

### **Pre-Session Checklist** - MUST COMPLETE BEFORE STARTING ANY STORY WORK

- [ ] Read and understand the complete story document including all acceptance criteria
- [ ] Identify all checklist items that will be addressed in this session
- [ ] Commit to updating documentation throughout the session (not just at the end)
- [ ] Plan the comprehensive work summary structure required at session end
- [ ] Verify understanding of all Definition of Done criteria

### **Session End Checklist** - MUST COMPLETE BEFORE ENDING ANY STORY SESSION

- [ ] All completed work is marked with `[x]` in story checklists
- [ ] Comprehensive work summary added with all required sections (Overview, Implementation Details, Key Highlights, Test Results, Architectural Alignment, Foundation for Next Phases, Quality Metrics, Deliverables Summary)
- [ ] All acceptance criteria explicitly addressed and confirmed complete
- [ ] All Definition of Done items verified and marked complete
- [ ] Agentic documents (AGENTS.md, copilot-instructions.md) updated with new patterns
- [ ] Status clearly marked as **COMPLETE** ✅ with next phase identified

### **Process Violation Recovery** - IMMEDIATE ACTION REQUIRED IF CHECKLIST MISSED

- [ ] **If story checklist not updated**: STOP all work, update immediately before proceeding
- [ ] **If work summary missing**: Session CANNOT end until comprehensive summary added
- [ ] **If agentic docs not updated**: All new patterns MUST be documented before session close
- [ ] **Acknowledge violation**: Add explicit note about process improvement for future sessions

**⚠️ CRITICAL**: Story documentation is as important as code implementation. NO EXCEPTIONS.

---

This document defines the **MVP requirements** for our bespoke RunPod worker. It is written to inform AI agents of the scope, constraints, and objectives.

## **Current Implementation Status**

### **Phase 1: Foundation Infrastructure - COMPLETE ✅**

- **MMI-003: Model Management Framework** - Complete model loading, caching, and memory management system
- **MMI-004: Basic Handler Routing** - Complete request routing infrastructure with validation, logging, and RunPod integration

### **Phase 2: Core Modality Implementation - IN PROGRESS**

- **MMI-005: FLUX.1 Text-to-Image Handler** - COMPLETE ✅ - Production-ready FLUX.1 Schnell fp8 text-to-image generation with <15 second inference times, comprehensive validation, and full integration with model management and routing systems
- **MMI-006: ControlNet Integration** - COMPLETE ✅ - Production-ready ControlNet guided image generation with Canny edge detection and depth estimation, <20 second inference times, memory-efficient shared component architecture
- **MMI-007: AnimateDiff Integration** - COMPLETE ✅ - Production-ready image-to-video generation with motion adapters, 16-frame output targeting <25 second inference times, comprehensive video processing pipeline with MP4/GIF/WebM support

### **Current Architecture: Request Routing System**

The worker now implements a sophisticated request routing infrastructure:

#### **Request Processing Pipeline**

1. **Request Reception**: RunPod serverless handler (`src/main.py`) receives requests
2. **Modality Detection**: Automatic detection using parameter signatures or explicit modality specification
3. **Request Validation**: Comprehensive parameter validation with detailed error reporting
4. **Handler Routing**: Dynamic routing to appropriate modality-specific handlers
5. **Model Management**: Intelligent loading and caching using MMI-003 ModelManager
6. **Response Formatting**: Standardized response format with RunPod compatibility

#### **Key Infrastructure Components**

- **BaseHandler** (`src/handlers/base_handler.py`): Abstract base class defining standard interface for all modality implementations
- **MultiModalHandler** (`src/handlers/multi_modal_handler.py`): Central routing system managing request lifecycle
- **RequestValidator** (`src/utils/request_validator.py`): Intelligent modality detection and parameter validation
- **ResponseFormatter** (`src/utils/response_formatter.py`): Standardized response formatting with error categorization
- **LoggingConfig** (`src/utils/logging_config.py`): Structured logging with request tracking and performance monitoring

#### **FLUX.1 Text-to-Image Implementation (MMI-005 Complete)**

- **FluxHandler** (`src/handlers/flux_handler.py`): Production-ready text-to-image handler with comprehensive parameter validation, performance tracking, and <15 second inference guarantee
- **FluxModel** (`src/models/flux_model.py`): FLUX.1 Schnell fp8 model wrapper with memory optimization (12-15GB target), HuggingFace diffusers integration, and intelligent caching
- **ImageProcessor** (`src/utils/image_utils.py`): Comprehensive image processing utilities with PIL/PyTorch integration, format conversion (PNG/JPEG/WebP), and base64 encoding
- **TextToImageSchema** (`src/schemas/text_to_image_schema.py`): Pydantic validation schemas with automatic dimension adjustment, parameter constraints, and error handling

#### **ControlNet Guided Generation Implementation (MMI-006 Complete)**

- **ControlNetHandler** (`src/handlers/controlnet_handler.py`): Production-ready guided image generation with Canny edge detection and depth estimation, comprehensive parameter validation, <20 second inference guarantee, and performance tracking
- **ControlNetModel** (`src/models/controlnet_model.py`): Memory-efficient ControlNet model wrapper with shared component architecture leveraging FLUX.1 base components (VAE, text encoder) for 16GB total target usage
- **Control Processors** (`src/utils/control_processors.py`): Robust image preprocessing with CannyProcessor (OpenCV-based edge detection) and DepthProcessor (MiDaS-based depth estimation) using extensible factory pattern
- **ControlNetSchema** (`src/schemas/controlnet_schema.py`): Comprehensive Pydantic validation for ControlNet requests with control-specific parameters, image validation, and detailed error handling

#### **AnimateDiff Image-to-Video Implementation (MMI-007 Complete)**

- **AnimateDiffHandler** (`src/handlers/animatediff_handler.py`): Production-ready image-to-video generation with motion adapter integration, comprehensive parameter validation, <25 second inference guarantee for 16-frame videos, and performance tracking
- **AnimateDiffModel** (`src/models/animatediff_model.py`): Memory-efficient AnimateDiff model wrapper with shared FLUX.1 component architecture (UNet, VAE, text encoders) targeting 16GB total usage and motion adapter integration
- **Video Processing Pipeline** (`src/utils/video_utils.py`): Comprehensive video encoding utilities with VideoEncoder (MP4/GIF/WebM support) and FrameProcessor (interpolation, validation) using imageio/FFmpeg integration
- **ImageToVideoSchema** (`src/schemas/image_to_video_schema.py`): Robust Pydantic validation for image-to-video requests with motion parameters (motion_bucket_id, noise_aug_strength), frame count constraints, and video metadata tracking

### **Next Phase: Modality Implementation (MMI-005 through MMI-010)**

Ready to implement specific modality handlers using established routing infrastructure.

---

## Project Goal

Create a proof-of-concept worker on RunPod that supports **multi-modal AI inference** with a compact model footprint. This MVP should:

- Cover **all required endpoint types** (text-to-image, image-to-image, text-to-video, image-to-video, video-to-video, inpainting, ControlNet, camera control).
- Stay **below 80 GB** in model storage.
- Demonstrate **end-to-end functionality** across modalities with minimal latency and manageable cost.

---

## MVP Requirements

### Modalities

- **Text→Image / Image→Image**

  - Model: `FLUX.1 [schnell] (fp8)` (~12–15 GB)

- **Control**

  - Models: `ControlNet (canny + depth only)` (~3–4 GB)

- **Image→Video**

  - Model: `AnimateDiff (1 motion adapter)` (~2 GB)

- **Text→Video**

  - Model: `LTX-Video 2B (distilled)` (~6–10 GB)

- **Video→Video**

  - Lightweight editor placeholder (optional for MVP)

- **Inpainting**

  - Model: `SDXL Inpainting` (~6–8 GB)

- **Camera Control**

  - Module: `CameraCtrl` (code + tiny weights, <1 GB)

### Total Footprint

- Expected usage: **~30–40 GB**
- Hard cap: **≤ 80 GB**

---

## Infrastructure Notes

- **Deployment Target:** RunPod (serverless for MVP)
- **Storage:** RunPod **Network Volume** (shared, avoids re-downloads)
- **Startup:** Enable **FlashBoot** to minimize cold-starts
- **Cache:** Centralize Hugging Face cache in `/runpod-volume/cache/hf` and prune regularly

---

## Scaling Path (NOT REQUIRED CURRENTLY)

- **Beta (≤ 150 GB)**: Add `SD3-Medium`, expand ControlNets, upgrade Text→Video to LTX-13B fp8, optionally add `HunyuanVideo`.
- **Production (300+ GB)**: Add WAN 2.2 AIO, multiple backbones, full ControlNet suite, multiple AnimateDiff modules, upscalers, and advanced V2V editors.

---

## Rules for AI Agents

### **Infrastructure and Resource Management**

1. Always verify **disk usage** before pulling new models.
2. Do not exceed **80 GB** total in MVP stage.
3. Ensure **all endpoints** remain functional.
4. Favor **fp8/distilled** weights where possible.
5. Prefer **pods** only if GPU utilization exceeds 40%; otherwise stay serverless.
6. Use the **directory layout** defined in `Runpod Worker Model Plan`.

### **Code Architecture and Implementation**

1. Follow the **established repository structure** - all code in `src/`, tests in `tests/`, Docker config in `docker/`.
2. Use the **MultiModalHandler** routing system for all request processing - do NOT bypass the established routing infrastructure.
3. Create **modality-specific handlers** in `src/handlers/` directory inheriting from BaseHandler abstract class.
4. Implement **comprehensive tests** for each new modality in both `tests/unit/` and `tests/integration/`.
5. Update **API documentation** in `docs/api.md` when adding new endpoints or modifying schemas.

### **Request Routing System (MMI-004) Patterns**

1. **Handler Implementation**: All modality handlers MUST inherit from BaseHandler and implement all abstract methods (`validate_request()`, `load_models()`, `process_inference()`, `format_response()`).
2. **Request Validation**: Use RequestValidator for parameter validation and modality detection - extend modality signatures for new parameter patterns.
3. **Response Formatting**: Use ResponseFormatter for all responses to ensure consistent structure and RunPod compatibility.
4. **Logging Integration**: Use structured logging from LoggingConfig with proper request context management and performance monitoring.
5. **Handler Registration**: Register new handlers with MultiModalHandler using `register_handler(modality, handler)` pattern.
6. **Error Handling**: Use established error categories (ValidationError, ModelError, InferenceError, etc.) with detailed error messages and suggestions.

### **Model Management Integration (MMI-003)**

1. **ModelManager Integration**: Use the global ModelManager instance for all model loading, caching, and memory management operations.
2. **BaseModel Implementation**: Create model classes inheriting from BaseModel with proper memory tracking and validation methods.
3. **Memory Management**: Follow established patterns for model loading, eviction, and memory optimization using MemoryMonitor.
4. **Configuration**: Use established config patterns for model paths, cache directories, and resource limits.

---

## Key Directory Structure (MVP)

```bash
/runpod-volume/models/
  flux/
    flux1-schnell-fp8.safetensors
  controlnet/
    canny.safetensors
    depth.safetensors
  animatediff/
    motion_adapter.safetensors
  video_backbones/
    ltx-2b-distilled/
  inpaint/
    sdxl-inpaint.safetensors
  camera/
    camctrllib/
```

---

## Success Criteria

- Worker can:

  - Generate an image from text (T2I)
  - Apply ControlNet guidance
  - Animate an image into video (I2V)
  - Generate a video from text (T2V)
  - Perform inpainting
  - Demonstrate camera control in video

- Total disk ≤ 80 GB
- Runs cost-effectively in serverless mode

---

## Development Guidelines for AI Agents

### Repository Structure (Implemented)

The foundation has been established following Python best practices:

```bash
workers/multi-model-worker/
├── src/                           # Source code directory
│   ├── __init__.py               # Package exports
│   ├── main.py                   # MultiModalHandler entry point
│   ├── handlers/                 # Modality-specific handlers
│   │   └── __init__.py          # Handler imports
│   ├── models/                   # Model wrapper classes
│   │   └── __init__.py          # Model imports
│   └── utils/                    # Shared utilities
│       └── __init__.py          # Utility imports
├── docker/                       # Docker configuration
│   ├── Dockerfile               # Multi-stage build
│   └── requirements.txt         # Python dependencies
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
└── docs/                        # Documentation
    ├── api.md                   # API reference
    └── deployment.md            # Deployment guide
```

### Implementation Patterns

#### Handler Implementation Pattern

When implementing new modality handlers, follow this structure:

```python
# src/handlers/{modality}_handler.py
class {Modality}Handler:
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = model_cache_dir
        self.model = None

    def load_model(self):
        """Load the required model for this modality"""
        pass

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request for this modality"""
        pass

    def unload_model(self):
        """Free model memory when not in use"""
        pass
```

#### Model Wrapper Pattern

Create model wrappers in `src/models/` for shared functionality:

```python
# src/models/{model}_model.py
class {Model}Model:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def load(self):
        """Load model with optimization"""
        pass

    def generate(self, **kwargs):
        """Generate output with model"""
        pass

    def optimize_memory(self):
        """Apply memory optimizations"""
        pass
```

#### Test Implementation Requirements

For each new modality, implement both unit and integration tests:

```python
# tests/unit/test_{modality}_handler.py
class Test{Modality}Handler(unittest.TestCase):
    def test_handler_initialization(self):
        pass

    def test_model_loading(self):
        pass

    def test_inference_processing(self):
        pass

# tests/integration/test_{modality}_integration.py
class Test{Modality}Integration(unittest.TestCase):
    def test_end_to_end_workflow(self):
        pass

    def test_error_handling(self):
        pass
```

### Development Workflow for AI Agents

1. **Phase-Based Implementation**: Follow the MMI story sequence (MMI-001 through MMI-012)
2. **Test-Driven Development**: Write tests before implementing functionality
3. **Memory Management**: Implement smart model loading/unloading for GPU memory optimization
4. **Documentation Updates**: Update `docs/api.md` with new endpoints and schemas
5. **Docker Optimization**: Update `docker/requirements.txt` as needed for new dependencies
6. **Story Progress Tracking**: Always update story document checklists as tasks are completed
7. **Comprehensive Documentation**: Add detailed work summaries to story documents upon completion

### Story Management Requirements

#### Mandatory Story Documentation Process

When working on any MMI story, AI assistants must:

1. **Update Story Checklists**: Mark each completed task with `[x]` in the story document
2. **Track Progress**: Update "Inspect and modify", "Tests", and "General" sections
3. **Complete Work Summaries**: Add comprehensive "Summary of Work Completed" sections
4. **Document Implementation**: Include directory structures, files, test results, and architectural notes
5. **Validate Deliverables**: Ensure all acceptance criteria and Definition of Done items are satisfied

#### Work Summary Structure

```markdown
## Summary of Work Completed

### Overview

[Brief summary with completion date]

### [Component] Implementation

[Detailed breakdown of major components]

### Key Implementation Highlights

[Technical achievements and important details]

### Test Results / Validation

[Test results and functionality verification]

### Architectural Alignment

[Integration notes and system alignment]

### Deliverables Summary

[Final checklist of completed items]

**Status**: **COMPLETE** ✅
**Next Phase**: [Follow-up work identification]
```

### Quality Standards

- **Test Coverage**: Minimum 90% coverage for all new code
- **Documentation**: All public methods must have docstrings
- **Error Handling**: Comprehensive error handling with proper logging
- **Memory Efficiency**: Implement model eviction when GPU memory is constrained
- **Performance**: Meet target inference times specified in strategy document

### Integration Points

- **Frontend Integration**: Handler responses must be compatible with Media Labs hooks architecture
- **RunPod Integration**: Follow RunPod serverless handler patterns in `main.py`
- **Model Storage**: Use established `/runpod-volume/models/` directory structure
- **Logging**: Use structured logging consistent with existing patterns

---

## Model Management Framework (Completed)

### Overview

A comprehensive **intelligent model management system** has been implemented to handle loading, caching, and eviction of AI models within GPU memory constraints. This framework provides the core infrastructure for multi-modal inference operations.

### Core Components

#### **ModelManager** (`src/models/model_manager.py`)

- **LRU Eviction**: Automatically evicts least recently used models when memory limits are reached
- **Thread Safety**: Full thread-safe operations using RLock and ThreadPoolExecutor
- **Registration System**: Dynamic model registration with metadata and configuration
- **Status Monitoring**: Real-time tracking of model states, usage statistics, and performance metrics

#### **MemoryMonitor** (`src/models/memory_monitor.py`)

- **Real-time Monitoring**: Continuous GPU and system memory tracking
- **Threshold Detection**: Configurable warning (75%) and eviction (85%) thresholds
- **Callback System**: Event-driven memory pressure notifications
- **Cache Clearing**: Automatic GPU cache management when memory pressure detected

#### **BaseModel** (`src/models/base_model.py`)

- **Abstract Interface**: Standardized model lifecycle (load/unload/infer)
- **LRU Scoring**: Priority-aware scoring for intelligent eviction decisions
- **Usage Tracking**: Access patterns and memory consumption monitoring
- **Status Reporting**: Comprehensive model state and performance information

#### **Configuration Management** (`src/utils/config.py`)

- **Environment Variables**: Configurable memory thresholds, model limits, cache directories
- **Validation**: Automatic configuration validation with sensible defaults
- **Test Environment**: Special handling for development and test environments

### Key Features

#### **Intelligent Memory Management**

```python
# Automatic eviction when memory limits reached
model_manager.register_model("flux1", FluxModel, model_path, estimated_memory_mb=4000)
model = model_manager.get_model("flux1")  # Loads with automatic eviction if needed
```

#### **Thread-Safe Operations**

```python
# Concurrent model loading across multiple threads
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(model_manager.get_model, f"model_{i}") for i in range(5)]
    models = [f.result() for f in futures]  # All models loaded safely
```

#### **Real-time Monitoring**

```python
# Memory monitoring with callbacks
memory_monitor.add_eviction_callback(lambda: model_manager.evict_lru_models(1))
stats = memory_monitor.get_current_stats()  # Current GPU/CPU memory usage
```

### Integration Patterns

#### **Handler Integration** (`src/main.py`)

- Seamless integration with existing `MultiModalHandler`
- Model registration during worker initialization
- Automatic model loading and eviction during inference

#### **Error Handling**

- **ModelLoadError**: Model loading failures with detailed context
- **MemoryError**: Insufficient memory conditions with suggested actions
- **ConcurrencyError**: Thread safety violations and deadlock prevention

#### **Performance Optimization**

- **Model Pooling**: Efficient reuse of loaded models across requests
- **Memory Efficiency**: Automatic cleanup and garbage collection triggers
- **Load Balancing**: Priority-based eviction protects critical models

### Testing and Validation

#### **Comprehensive Test Suite**

- **Unit Tests**: 29 passing tests covering all components (tests/unit/)
- **Integration Tests**: End-to-end lifecycle validation (tests/integration/)
- **Performance Benchmarks**: Memory efficiency and throughput testing
- **Framework Validation**: `validate_framework.py` - working end-to-end validation

#### **Test Coverage**

- ✅ **ModelManager**: Registration, loading, eviction, thread safety, status monitoring
- ✅ **MemoryMonitor**: Stats collection, threshold detection, callback system
- ✅ **BaseModel**: Lifecycle management, LRU scoring, usage tracking
- ✅ **Configuration**: Environment handling, validation, test compatibility

### Development Workflow

#### **Adding New Model Types**

1. Extend `BaseModel` with specific model implementation
2. Register with `ModelManager` during initialization
3. Implement `load()`, `unload()`, `infer()` methods
4. Configure memory requirements and priority

#### **Memory Management Rules**

- **Warning Threshold**: 75% memory usage triggers proactive cleanup
- **Eviction Threshold**: 85% memory usage forces model eviction
- **Priority Protection**: High-priority models (>70) protected from eviction
- **LRU Algorithm**: Least recently used models evicted first

#### **Error Handling Patterns**

```python
try:
    model = model_manager.get_model("flux1")
    result = model.infer(inputs)
except ModelLoadError as e:
    # Handle loading failures - model unavailable or corrupted
except MemoryError as e:
    # Handle memory pressure - suggest model eviction or smaller batch
except ConcurrencyError as e:
    # Handle thread contention - retry with backoff
```

### Current Status (Phase 1 & 2 Complete)

✅ **MMI-001 Complete**: Repository structure and foundation established
✅ **MMI-002 Complete**: Model management framework implemented
✅ **MMI-005 Complete**: FLUX.1 Text-to-Image Handler production-ready
✅ **MMI-006 Complete**: ControlNet Integration with guided image generation
✅ **MMI-007 Complete**: AnimateDiff Integration with image-to-video generation
🔄 **Next**: Text-to-Video Generation (MMI-008)

The multi-modal infrastructure is production-ready with comprehensive text-to-image, guided image generation, and image-to-video capabilities. Ready for direct text-to-video implementation and additional modalities.
