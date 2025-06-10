CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -g

# Build directory
BUILD_DIR = /tmp/build
BUILD_DIR_ABS = $(abspath $(BUILD_DIR))

# Source files (absolute paths)
HEADERS_ABS = $(abspath ../advanced_bind_from_value_to_type.hpp)
DEMO_SRC_ABS = $(abspath ./advanced_bind_demo.cpp)
SIMPLE_SRC_ABS = $(abspath ./simple_demo.cpp)
BASE_CRTP_SRC_ABS = $(abspath ../base_crtp_demo.cpp)
ENHANCED_SRC_ABS = $(abspath ../enhanced_crtp_trait_bind_demo.cpp)

# Target executables (absolute paths)
DEMO_TARGET_ABS = $(BUILD_DIR_ABS)/advanced_demo
SIMPLE_TARGET_ABS = $(BUILD_DIR_ABS)/simple_demo
BASE_TARGET_ABS = $(BUILD_DIR_ABS)/base_demo
ENHANCED_TARGET_ABS = $(BUILD_DIR_ABS)/enhanced_demo

# Compatibility with relative paths
HEADERS = ../advanced_bind_from_value_to_type.hpp
DEMO_SRC = ./advanced_bind_demo.cpp
SIMPLE_SRC = ./simple_demo.cpp
BASIC_SRC = ../base_crtp_demo.cpp ../enhanced_crtp_trait_bind_demo.cpp
DEMO_TARGET = $(BUILD_DIR)/advanced_demo
SIMPLE_TARGET = $(BUILD_DIR)/simple_demo
BASIC_TARGETS = $(BUILD_DIR)/base_demo $(BUILD_DIR)/enhanced_demo

# Default target
.PHONY: all clean demo simple basic help clean-build

all: simple demo basic

# Ensure clean build directory
clean-build:
	@echo "Setting up clean build environment..."
	@if [ -d "$(BUILD_DIR_ABS)" ]; then \
		echo "Removing existing build directory: $(BUILD_DIR_ABS)"; \
		rm -rf $(BUILD_DIR_ABS); \
	fi
	@echo "Creating fresh build directory: $(BUILD_DIR_ABS)"
	@mkdir -p $(BUILD_DIR_ABS)

# Simple demo (recommended first test)
simple: clean-build $(SIMPLE_TARGET)

$(SIMPLE_TARGET): $(SIMPLE_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(SIMPLE_TARGET_ABS) $(SIMPLE_SRC_ABS)

# Advanced demo
demo: clean-build $(DEMO_TARGET)

$(DEMO_TARGET): $(DEMO_SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(DEMO_TARGET_ABS) $(DEMO_SRC_ABS)

# Basic demos
basic: clean-build $(BASIC_TARGETS)

$(BUILD_DIR)/base_demo: ../base_crtp_demo.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(BASE_TARGET_ABS) $(BASE_CRTP_SRC_ABS)

$(BUILD_DIR)/enhanced_demo: ../enhanced_crtp_trait_bind_demo.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(ENHANCED_TARGET_ABS) $(ENHANCED_SRC_ABS)

# Test targets
test: simple
	@echo "=== Running Simple Value-to-Type Binding Demo ==="
	@./$(SIMPLE_TARGET)
	@echo ""
	@echo "=== Demo completed successfully! ==="

test-advanced: demo
	@echo "=== Running Advanced Value-to-Type Binding Demo ==="
	@./$(DEMO_TARGET)
	@echo ""
	@echo "=== Advanced demo completed successfully! ==="

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts from: $(BUILD_DIR_ABS)"
	@rm -rf $(BUILD_DIR_ABS)
	@rm -f $(abspath *.o)
	@echo "Clean completed."

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all demos in clean build directory"
	@echo "  simple       - Build simple demo (recommended first test)"
	@echo "  demo         - Build advanced value-binding demo"
	@echo "  basic        - Build basic CRTP demos"
	@echo "  test         - Run the simple demo"
	@echo "  test-advanced- Run the advanced demo"
	@echo "  clean        - Remove build directory and artifacts"
	@echo "  clean-build  - Set up clean build environment"
	@echo "  help         - Show this help message"
	@echo ""
	@echo ""
	@echo "Absolute Project Paths:"
	@echo "  Current Dir: $(abspath .)"
	@echo "  Build Dir:   $(BUILD_DIR_ABS)"
	@echo "  Header:      $(HEADERS_ABS)"
	@echo "  Simple Src:  $(SIMPLE_SRC_ABS)"
	@echo "  Demo Src:    $(DEMO_SRC_ABS)"
	@echo "  Base Src:    $(BASE_CRTP_SRC_ABS)"
	@echo "  Enhanced:    $(ENHANCED_SRC_ABS)"
	@echo ""
	@echo "Advanced Value-to-Type Binding System"
	@echo "Author: Shen Xiaolong (申晓龙)"
	@echo "Repository: https://github.com/shenxiaolong-code/mlir-crtp-proposal" 