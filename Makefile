# =========================
# Compiler settings
# =========================
CXX       := g++
CXXFLAGS  := -std=c++17 -O2 -Wall -Wextra -pedantic -Ihead
LDFLAGS   :=

# =========================
# Project structure
# =========================
TARGET    := main

SRC_DIR   := src
INC_DIR   := head
BUILD_DIR := build

MAIN_SRC  := main.cpp
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
ALL_SRC   := $(MAIN_SRC) $(SRC_FILES)

OBJ_FILES := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(ALL_SRC))

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(CXX) $(OBJ_FILES) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(TARGET)
	./$(TARGET)

.PHONY: debug
debug: CXXFLAGS := -std=c++17 -O0 -g -Wall -Wextra -pedantic -Ihead
debug: clean all

.PHONY: release
release: CXXFLAGS := -std=c++17 -O3 -DNDEBUG -Wall -Wextra -pedantic -Ihead
release: clean all

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: rebuild
rebuild: clean all