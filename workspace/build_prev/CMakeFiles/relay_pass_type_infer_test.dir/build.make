# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/build

# Include any dependencies generated for this target.
include CMakeFiles/relay_pass_type_infer_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/relay_pass_type_infer_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/relay_pass_type_infer_test.dir/flags.make

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o: CMakeFiles/relay_pass_type_infer_test.dir/flags.make
CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o: ../tests/cpp/relay_pass_type_infer_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o -c /workspace/tests/cpp/relay_pass_type_infer_test.cc

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/tests/cpp/relay_pass_type_infer_test.cc > CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.i

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/tests/cpp/relay_pass_type_infer_test.cc -o CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.s

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.requires:

.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.requires

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.provides: CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.requires
	$(MAKE) -f CMakeFiles/relay_pass_type_infer_test.dir/build.make CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.provides.build
.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.provides

CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.provides.build: CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o


# Object files for target relay_pass_type_infer_test
relay_pass_type_infer_test_OBJECTS = \
"CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o"

# External object files for target relay_pass_type_infer_test
relay_pass_type_infer_test_EXTERNAL_OBJECTS =

relay_pass_type_infer_test: CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o
relay_pass_type_infer_test: CMakeFiles/relay_pass_type_infer_test.dir/build.make
relay_pass_type_infer_test: libtvm.so
relay_pass_type_infer_test: /usr/lib/libgtest.a
relay_pass_type_infer_test: CMakeFiles/relay_pass_type_infer_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable relay_pass_type_infer_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/relay_pass_type_infer_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/relay_pass_type_infer_test.dir/build: relay_pass_type_infer_test

.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/build

CMakeFiles/relay_pass_type_infer_test.dir/requires: CMakeFiles/relay_pass_type_infer_test.dir/tests/cpp/relay_pass_type_infer_test.cc.o.requires

.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/requires

CMakeFiles/relay_pass_type_infer_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/relay_pass_type_infer_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/clean

CMakeFiles/relay_pass_type_infer_test.dir/depend:
	cd /workspace/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace /workspace /workspace/build /workspace/build /workspace/build/CMakeFiles/relay_pass_type_infer_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/relay_pass_type_infer_test.dir/depend

