# =============================================================================
# CIFAR-10 Dataset Loader for Godot
# =============================================================================
# 
# Runtime loader for CIFAR-10 image datasets in training loops.
# Loads individual images or batches on-demand for machine learning.
#
# Features:
# - Configurable paths and labels file
# - Single image and batch loading
# - Image format validation and debugging
# - Optimized texture-to-array conversion
#
# Author: zephyrus @ inchwormgames
# Version: 1.0.0
# License: MIT
#
# =============================================================================

class_name CIFARLoader
extends Node

# =============================================================================
# CONFIGURATION
# =============================================================================

@export_group("Paths")
@export var image_dir: String = "res://cifar_data/"
## Directory containing PNG files and labels.json
@export var labels_file: String = "labels.json"
## Name of the labels JSON file

var labels_dict = {}

func _ready():
	# Load the image ID -> label mapping
	var file = FileAccess.open(image_dir + labels_file, FileAccess.READ)
	if file:
		var json_string = file.get_as_text()
		file.close()
		
		var json = JSON.new()
		var parse_result = json.parse(json_string)
		if parse_result == OK:
			var json_data = json.get_data()
			labels_dict = json_data["labels"]
		else:
			print("Error parsing JSON: ", json.get_error_message())
	
	# Test function available but not auto-run
	# Call test_loader() manually if needed

func texture_to_array(texture: Texture2D) -> Array:
	# Convert Godot texture to 32x32x3 RGB array (OPTIMIZED VERSION)
	var img = texture.get_image()
	
	# Make sure it's 32x32 RGB
	if img.get_width() != 32 or img.get_height() != 32:
		print("Warning: Image is not 32x32, resizing...")
		img.resize(32, 32)
	
	if img.get_format() != Image.FORMAT_RGB8:
		img.convert(Image.FORMAT_RGB8)
	
	# OPTIMIZATION: Use raw byte access instead of get_pixel() calls
	var raw_bytes = img.get_data()
	var data = []
	data.resize(32)
	
	# Pre-allocate all rows for performance
	for y in range(32):
		var row = []
		row.resize(32)
		data[y] = row
	
	# Process raw bytes directly - eliminates 1,024 function calls!
	var byte_index = 0
	for y in range(32):
		for x in range(32):
			var r = raw_bytes[byte_index] / 255.0
			var g = raw_bytes[byte_index + 1] / 255.0
			var b = raw_bytes[byte_index + 2] / 255.0
			
			data[y][x] = [r, g, b]  # RGB as 0.0-1.0
			byte_index += 3
	
	return data

func load_single_image(image_id: String) -> Dictionary:
	# Load one PNG image and return its data + label
	var img_path = image_dir + image_id + ".png"
	
	if not FileAccess.file_exists(img_path):
		print("Image not found: ", img_path)
		return {}
	
	var texture = load(img_path) as Texture2D
	if not texture:
		print("Failed to load texture: ", img_path)
		return {}
	
	var image_data = texture_to_array(texture)
	var label = labels_dict.get(image_id, "unknown")
	
	return {
		"image": image_data,
		"label": label,
		"id": image_id
	}

func load_batch(start_id: int, batch_size: int) -> Dictionary:
	# Load multiple images in sequence for training
	# Note: For JSON batches, use load_batch_from_json() instead
	var images = []
	var labels = []
	var ids = []
	
	for i in range(batch_size):
		var image_id = str(start_id + i)
		var result = load_single_image(image_id)
		
		if result.has("image"):
			images.append(result.image)
			labels.append(result.label)
			ids.append(result.id)
		else:
			print("Skipping image: ", image_id)
	
	return {
		"images": images,
		"labels": labels, 
		"ids": ids,
		"count": images.size()
	}

func load_batch_from_json(batch_file_path: String) -> Dictionary:
	# Load a batch from a preprocessed JSON file (much faster than PNG loading)
	if not FileAccess.file_exists(batch_file_path):
		print("Batch file not found: ", batch_file_path)
		return {}
	
	var file = FileAccess.open(batch_file_path, FileAccess.READ)
	if not file:
		print("Failed to open batch file: ", batch_file_path)
		return {}
	
	var json_string = file.get_as_text()
	file.close()
	
	var json = JSON.new()
	var parse_result = json.parse(json_string)
	if parse_result != OK:
		print("Error parsing batch JSON: ", json.get_error_message())
		return {}
	
	var batch_data = json.get_data()
	var images = []
	var labels = []
	var ids = []
	
	for image_id in batch_data.keys():
		var item = batch_data[image_id]
		images.append(item.data)
		labels.append(item.label)
		ids.append(image_id)
	
	return {
		"images": images,
		"labels": labels,
		"ids": ids,
		"count": images.size()
	}

func debug_image_stats(image_data: Array):
	# Print useful info about loaded image data
	if image_data.size() != 32:
		print("Error: Image height is not 32, got: ", image_data.size())
		return
	
	if image_data[0].size() != 32:
		print("Error: Image width is not 32, got: ", image_data[0].size())
		return
	
	if image_data[0][0].size() != 3:
		print("Error: Image channels is not 3, got: ", image_data[0][0].size())
		return
	
	# Show some sample pixels
	var top_left = image_data[0][0]
	var center = image_data[16][16]
	
	print("Image format: 32x32x3 ✓")
	print("Top-left pixel RGB: ", top_left)
	print("Center pixel RGB: ", center)
	print("Value range (should be 0.0-1.0):")
	print("  R: ", top_left[0], " to ", center[0])
	print("  G: ", top_left[1], " to ", center[1]) 
	print("  B: ", top_left[2], " to ", center[2])

# =============================================================================
# TESTING
# =============================================================================

func test_loader():
	# Test loading a single image and a small batch
	print("Testing CIFAR loader...")
	
	var result = load_single_image("1")
	if result.has("image"):
		print("Successfully loaded image 1, label: ", result.label)
		debug_image_stats(result.image)
	else:
		print("Failed to load test image")
	
	print("\nTesting batch load...")
	var batch = load_batch(1, 3)
	print("Loaded batch with ", batch.count, " images")
	print("Labels: ", batch.labels)
