# - Try to find DNNL
#
# The following variables are optionally searched for defaults
#  DNNL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  DNNL_FOUND          : set to true if dnnl is found.
#  DNNL_INCLUDE_DIR    : path to dnnl include dir.
#  DNNL_LIBRARIES      : list of libraries for dnnl

if(DNNL_INCLUDE_DIRS)
  # Already in cache, be silent
  set (DNNL_INCLUDE_DIRS_FIND_QUIETLY TRUE)
endif (DNNL_INCLUDE_DIRS)

find_path(DNNL_ROOT_DIR
  NAMES include/dnnl.hpp
  HINTS /usr/local/ $ENV{DNNL_ROOT}
  DOC "dnnl root directory.")

find_path(_DNNL_INCLUDE_DIRS
  NAMES dnnl.hpp
  HINTS ${DNNL_ROOT_DIR}/include
  DOC "dnnl Include directory")

find_library(_DNNL_LIBRARY
  NAMES dnnl
  HINTS ${DNNL_ROOT_DIR}/lib ${DNNL_ROOT_DIR}/lib64
  # PATHS ${DNNL_ROOT_DIR}/lib
  # PATH_SUFFIXES so
  DOC "DNNL lib directory")

SET(DNNL_INCLUDE_DIR ${_DNNL_INCLUDE_DIRS})
SET(DNNL_LIBRARIES ${_DNNL_LIBRARY})

message(STATUS "${DNNL_INCLUDE_DIR}")
# handle the QUIETLY and REQUIRED arguments and set DNNL_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DNNL DEFAULT_MSG DNNL_LIBRARIES DNNL_INCLUDE_DIR)
MARK_AS_ADVANCED(DNNL_LIBRARIES DNNL_INCLUDE_DIR)
