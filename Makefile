# Assumes the libcudf-tutorial conda environment is activated:
#   conda activate libcudf-tutorial

CONDA_ENV_NAME = libcudf-tutorial

.DEFAULT_GOAL := all

BUILD_DIR  ?= build
CACHE_FILE  = $(BUILD_DIR)/CMakeCache.txt
CMAKE_ARGS  = -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) \
              -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
              -GNinja

.PHONY: all configure clean check-conda

# Correct conda environment should be activated before building.
check-conda:
	@if [ "$(CONDA_DEFAULT_ENV)" != "$(CONDA_ENV_NAME)" ]; then \
		echo "Error: conda environment '$(CONDA_ENV_NAME)' is not activated."; \
		echo "Run: conda activate $(CONDA_ENV_NAME)"; \
		exit 1; \
	fi

# Only run cmake configure when the cache is missing (first build or after clean)
all: check-conda $(CACHE_FILE)
	cmake --build $(BUILD_DIR)

$(CACHE_FILE):
	cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

configure: check-conda
	cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

clean:
	rm -rf $(BUILD_DIR)
