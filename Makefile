# Assumes the libcudf-env conda environment is activated:
#   conda activate libcudf-env

BUILD_DIR  ?= build
CACHE_FILE  = $(BUILD_DIR)/CMakeCache.txt
CMAKE_ARGS  = -DCMAKE_PREFIX_PATH=$(CONDA_PREFIX) \
              -DCMAKE_BUILD_TYPE=Release \
              -GNinja

.PHONY: all configure clean

# Only run cmake configure when the cache is missing (first build or after clean)
all: $(CACHE_FILE)
	cmake --build $(BUILD_DIR)

$(CACHE_FILE):
	cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

configure:
	cmake -B $(BUILD_DIR) $(CMAKE_ARGS)

clean:
	rm -rf $(BUILD_DIR)
