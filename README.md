# CIFAR-10 Dataset Preprocessor for Godot

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Godot Engine](https://img.shields.io/badge/Godot-4.0+-blue.svg)](https://godotengine.org/)
[![Performance](https://img.shields.io/badge/Performance-117%2B%20img%2Fsec-green.svg)](#performance)

A fast image preprocessing tool for converting PNG datasets into optimized batches for machine learning training. Built while exploring AI systems in Godot - turns out the engine handles dataset processing surprisingly well with some optimization work.

Started as a CIFAR-10 preprocessor, evolved into a general-purpose tool with memory pooling and thermal management.

## Why This Exists

Built this while exploring AI systems in Godot. Needed to preprocess CIFAR-10 datasets and was surprised how well Godot handled it with some optimization work. Figured others might find it useful.

## What It Does

- **117+ images/second** processing on laptop hardware (Intel Iris Xe)
- **<6MB RAM growth** during 50,000 image processing 
- **Thermal-safe** - won't overheat your laptop
- **Memory optimized** - eliminates 1,000+ allocations per image
- **Reliable** - handles errors and monitors performance

> **Note:** Benchmarked on laptop hardware. Desktop systems will be significantly faster.

## Features

### Performance & Optimization
- **Vectorized RGB conversion** - processes multiple pixels at once
- **Memory pooling** - reuses arrays to avoid garbage collection
- **PackedFloat32Array** - eliminates nested array allocations
- **Raw byte access** - skips slow pixel-by-pixel operations
- **Adaptive scaling** - adjusts workload to prevent overheating

### Thermal Management
- **Laptop-safe** - won't cook your hardware
- **CPU monitoring** - automatically throttles when needed
- **Desktop vs laptop modes** - different limits for different hardware
- **Sustained processing** - runs for hours without overheating

### Monitoring & Reliability
- **Performance metrics** - tracks speed and detects slowdowns
- **Memory monitoring** - cleans up before running out of RAM
- **Progress reporting** - shows ETA and current status
- **Error handling** - recovers from common issues
- **Debug functions** - tools for troubleshooting

### Output Formats
- **JSON batches** - human-readable
- **Binary format** - faster and smaller
- **Configurable batch sizes** - 1K to 10K+ images per file
- **Float32 precision** - lossless quality

## Use Cases

- **Preprocessing** - Convert PNG datasets to optimized batches for ML training
- **Loading** - Runtime loading of processed data for neural networks

## Quick Start

### Prerequisites
- **Godot 4.0+** (tested with 4.5+)
- **8GB+ RAM** recommended for large datasets
- **Multi-core CPU** for optimal performance
- **SSD storage** recommended for best I/O performance

### Installation
1. **Clone this repository** or download the scripts
   ```bash
   git clone https://github.com/Andraszao/Image-Preprocessor
   ```

2. **Add scripts to your Godot project**
   - Copy `cifar_preprocessor.gd` and `cifar_loader.gd` to your project
   - Create a new scene with a Node
   - Attach `cifar_preprocessor.gd` to the node

3. **Configure dataset path**
   - Set `dataset_path` in the editor to your PNG folder
   - Set `output_path` for processed batch files

4. **Run preprocessing**
   ```gdscript
   # Automatic processing
   preprocessor.process_on_ready = true
   preprocessor.auto_process_full_dataset = true
   
   # Or manual processing
   preprocessor.process_50k_dataset()
   ```

## Usage

### Basic Setup

1. Place `cifar_preprocessor.gd` and `cifar_loader.gd` in your project
2. Create a new scene with a Node and attach `cifar_preprocessor.gd`
3. Set the `dataset_path` to your CIFAR PNG folder
4. Configure `output_path` for where to save JSON batches

### Processing Options

- **Auto-process on startup**: Enable `auto_process_full_dataset` and `process_on_ready`
- **Manual processing**: Call `process_50k_dataset()` from code
- **Custom paths**: Use `process_custom_path("your/path/here")`

### Performance Tuning

- **Batch size**: Larger batches = faster processing, more RAM usage
- **Workload intensity**: Auto-detects optimal processing intensity, prevents CPU overheating
- **Memory management**: Configurable garbage collection frequency
- **Progress reporting**: Customizable progress update intervals

## Configuration

### Export Variables

```gdscript
# Dataset Paths
dataset_path: String = ""           # Path to your PNG dataset folder
output_path: String = "res://cifar_processed/"  # Where to save JSON batches

# Auto-Processing
auto_process_full_dataset: bool = false  # Start processing automatically
process_on_ready: bool = false           # Only works if auto_process is enabled

# Performance Tuning
batch_size: int = 2000                  # Images per output file (larger = faster, more RAM)
gc_frequency: int = 5                   # Garbage collection every N batches (lower = more frequent)
yield_frequency: int = 50               # Pause every N images for UI responsiveness (lower = smoother UI)

# Performance Scaling
max_workload_intensity: int = 0         # 0 = auto-detect, -1 = single-threaded
auto_scaling: bool = true               # Adjust workload based on CPU usage
thermal_throttle: bool = true           # Reduce intensity if CPU gets hot
target_cpu_usage: float = 75.0          # Target CPU usage percentage
laptop_mode: bool = true                 # Conservative thermal management for laptops
max_sustained_cpu: float = 60.0         # Max sustained CPU for laptops (prevents overheating)

# Debug & Performance
verbose_logging: bool = false           # Show detailed processing logs (impacts performance)
progress_report_interval: int = 100     # How often to report progress (in images)
max_memory_mb: float = 4000.0           # Max memory before forcing GC (0 = disabled)
memory_check_interval: int = 10         # Check memory pressure every N images
use_binary_format: bool = false         # Use binary format instead of JSON (recommended: true)
performance_validation: bool = true     # Enable performance regression detection
baseline_fps: float = 92.6             # Baseline FPS for performance comparison (auto-detected)
```

## Output Format

The preprocessor creates JSON batch files with the following structure:

### JSON Format (readable)
```json
{
  "1": {
    "id": "1",
    "data": [0.123, 0.456, 0.789, ...],  // 3072 floats (32x32x3 RGB)
    "label": "airplane",
    "convert_time_ms": 15,
    "format": "packed_float32"
  },
  "2": {
    "id": "2", 
    "data": [0.234, 0.567, 0.890, ...],
    "label": "automobile",
    "convert_time_ms": 12,
    "format": "packed_float32"
  }
}
```

### Binary Format (Production)
- **50-70% smaller files** than JSON
- **2-3x faster** loading and saving
- **Identical data quality** with float32 precision
- **Cross-platform compatible** binary format
```

## Loading Data

Use `cifar_loader.gd` to load processed data:

```gdscript
# Basic usage
var loader = CIFARLoader.new()

# Load a single image by ID
var image_data = loader.load_single_image("1")
print("Loaded image with label: ", image_data.label)

# Load a batch of images (from PNG files - slower)
var batch = loader.load_batch(1, 5)  # Load images 1-5

# Load from preprocessed JSON/binary (much faster)
var json_batch = loader.load_batch_from_json("res://cifar_processed/cifar_batch_0000.json")
print("Loaded ", json_batch.size(), " images from batch file")

# Advanced: Load with progress callback
var large_batch = loader.load_batch_with_progress(1, 1000, _on_load_progress)

func _on_load_progress(current: int, total: int):
    print("Loading progress: ", current, "/", total)
```

## Performance Guide

### Optimization Tips

| Setting | Recommended | Impact |
|---------|-------------|--------|
| **Batch Size** | 2000-5000 | Larger = faster processing, more RAM |
| **Laptop Mode** | `true` for laptops | Prevents thermal throttling |
| **Binary Format** | `true` for production | 50%+ faster saves, smaller files |
| **GC Frequency** | 5 batches | Prevents memory bloat |
| **Memory Limit** | 4000MB | Triggers cleanup before OOM |

### Expected Performance

| Hardware | Processing Speed | Memory Usage | Time (50K images) |
|----------|------------------|--------------|-------------------|
| **Laptop** (4-8 cores) | 100-150 img/sec | 70-100MB | 6-8 minutes |
| **Desktop** (8+ cores) | 150-250 img/sec | 80-120MB | 3-6 minutes |
| **High-end** (16+ cores) | 200-300+ img/sec | 90-150MB | 2-4 minutes |

### Troubleshooting

**Slow Performance?**
- Enable binary format (`use_binary_format = true`)
- Increase batch size to 5000+
- Disable verbose logging
- Use SSD storage

**High Memory Usage?**
- Reduce batch size to 1000-2000
- Lower GC frequency to 3
- Enable memory pressure monitoring

**CPU Overheating?**
- Enable laptop mode
- Reduce `max_sustained_cpu` to 50%
- Lower `target_cpu_usage` to 60%

## System Requirements

### Minimum Requirements
- **Godot Engine 4.0+**
- **4GB RAM** (for small datasets <10K images)
- **Dual-core CPU** (single-threaded fallback available)
- **HDD storage** (slower but functional)

### Recommended Setup
- **Godot Engine 4.5+** (latest stable)
- **8GB+ RAM** (for optimal batch processing)
- **Quad-core+ CPU** (enables adaptive scaling)
- **SSD storage** (significantly faster I/O)
- **Good cooling** (for sustained high-performance processing)

### Tested Platforms
- **Windows 10/11** (Primary development platform)

### Performance Notes
- **Intel Iris Xe Graphics**: 117+ img/sec sustained

## Contributing

We welcome contributions! Here's how you can help:

- **Report bugs** via GitHub Issues
- **Suggest features** or optimizations
- **Improve documentation**
- **Submit performance improvements**
- **Add benchmarks** for different hardware

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Test with `debug_single_image_processing()` and `benchmark_memory_allocation_fix()`
4. Submit a pull request with performance metrics

## Changelog

### v1.0.0 (Current)
- Initial release with full optimization suite
- 117+ img/sec sustained processing
- Thermal-aware processing for laptops
- Memory pooling eliminates allocation storms
- Real-time performance monitoring

## License

**MIT License** - Feel free to use in commercial and open-source projects!

See [LICENSE](LICENSE) for full details.

## Author

**zephyrus @ inchworm games**

- Website: [inchwormgames.com](https://inchwormgames.com)

---

<div align="center">

*If this project helped you, consider giving it a star!*

</div>
