
project(${CMAKE_SOURCE_DIR})
cmake_minimum_required(VERSION 3.5)


# load some user options
include("UserSettings.txt")


# ---------------------------------------------- now the technical part -----------------------------------------------


# find packages
find_package(CUDA 7.0 REQUIRED)
find_package(OpenMP)

if(CMAKE_VERSION VERSION_LESS "3.10")
  find_package(Matlab COMPONENTS MAIN_PROGRAM)
  matlab_get_version_from_matlab_run(${Matlab_MAIN_PROGRAM} matlab_version)
  if(${matlab_version})
    if(matlab_version VERSION_LESS "8.2")
      message("-- Could NOT find Matlab: Found unsuitable version \"${matlab_version}\", but required is at least \"8.2\" (${Matlab_MAIN_PROGRAM})")
      set(MATLAB_FOUND FALSE)
    else()
      message("-- Found Matlab: Found suitable version \"${matlab_version}\", minimum required is \"8.2\"")
    endif()
  else()
    message("-- Found Matlab: Found presumably suitable version, minimum required is \"8.2\"")
  endif()
else()
  find_package(Matlab 8.2)
endif()


# remove . and , so that it can be used in the compiler flag
string(REPLACE "." "" computeCapability ${computeCapability})
string(REPLACE "," "" computeCapability ${computeCapability})
set(arch_flag ${computeCapability})

# set(CUDA_HOST_COMPILER "path/to/compiler")

list(FIND ThrustArchitectures "CPP" useCPP)
list(FIND ThrustArchitectures "OMP" useOMP)
if(NOT OPENMP_FOUND)
  set(useOMP "-1")
  list(REMOVE_ITEM ThrustArchitectures "OMP")
endif()
list(FIND ThrustArchitectures "TBB" useTBB)
list(FIND ThrustArchitectures "CUDA" useCUDA)

# setting the compiler flags for use of OpenMP
if(NOT ${useOMP} EQUAL -1)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set the correct compute capability and enable C++11-standard, optimization level 3,
# in case of cuda version 7.0 a flag suppressing some "wrong" const warnings,
# and a flag allowing shared library creation
if(${CUDA_VERSION_STRING} STREQUAL "7.0")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--relaxed-constexpr)
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_${arch_flag};-std=c++14;-O3;-Xcompiler;-march=native;-Xcompiler;-fPIC)

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-lm -lc")

# include directories with all the header files
include_directories(${CMAKE_SOURCE_DIR}/src/include)

# make shared object file (libB12_shared.so) usable after install
get_filename_component(B12Version ${CMAKE_SOURCE_DIR} NAME)
set(B12InstallPath "${installPath}/B12/${B12Version}")
set(CMAKE_INSTALL_RPATH "${B12InstallPath}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# append Identity to Maps
string(REPLACE ";" " " temp_dim "${Dimensions}") # all dimensions to a string

set(temp_construct "")
foreach(dimt_it ${Dimensions})
  string(APPEND temp_construct "0 ")
endforeach()
string(STRIP ${temp_construct} temp_construct) # a string with (nr. of dimensions) zeros

set(Identity "${temp_dim}" "${temp_construct}")

list(INSERT Maps 0 "Identity")


# get maximal indices in the lists
list(LENGTH ThrustArchitectures nr_archs)
math(EXPR archs_max_index ${nr_archs}-1)

list(LENGTH MaxDepths nr_maxDepths)
math(EXPR maxDepths_max_index ${nr_maxDepths}-1)

list(LENGTH Grids nr_grids)
math(EXPR grids_max_index ${nr_grids}-1)

list(LENGTH Maps nr_maps)
math(EXPR maps_max_index ${nr_maps}-1)




# preprocessor macros for the Matlab files
set(ALL_N_PARAMS "")
set(ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS "")
set(ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_DIMS2_REALS2_MAXDEPTHS2 "")
set(ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS "")
set(ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS "")
set(ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS "")

set(max_n_params 2)

list(GET Dimensions 0 FIRST_DIM)
list(GET Reals 0 FIRST_REAL)
list(GET MaxDepths 0 FIRST_MAXDEPTH)


# include header that wraps all maps; only for Matlab files
set(include_maps_string "\n")
foreach(map ${Maps})
  string(APPEND include_maps_string "#include \"Map_${map}.h\"\n")
endforeach(map)

if(EXISTS ${CMAKE_SOURCE_DIR}/src/include/Maps.h)
  file(READ ${CMAKE_SOURCE_DIR}/src/include/Maps.h include_maps_file)
else()
  set(include_maps_file "")
endif()

if(NOT "${include_maps_string}" STREQUAL "${include_maps_file}")
  file(WRITE ${CMAKE_SOURCE_DIR}/src/include/Maps.h "${include_maps_string}")
endif()



# to avoid "relocation truncated to fit" errors due to too large shared object files, several sublibraries are created
set(libList "")



# explicitly instantiate all possible copy constructors of class ImplicitBoxTree<${thrustarch},${dim}>

set(explicit_Instant_string "")

foreach(thrustarch_it RANGE ${archs_max_index})
  list(GET ThrustArchitectures ${thrustarch_it} thrustarch)
  foreach(dim ${Dimensions})
    foreach(real ${Reals})
      foreach(maxDepth ${MaxDepths})
        foreach(thrustarch_it2 RANGE ${archs_max_index})
          list(GET ThrustArchitectures ${thrustarch_it2} thrustarch2)
          foreach(dim2 ${Dimensions})
            foreach(real2 ${Reals})
              foreach(maxDepth2 ${MaxDepths})
                
                configure_file(src/include/explicit_Instant_copy_constructors.hpp.in
                               ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.hpp.out @ONLY)
                file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.hpp.out
                     EXPLICIT_INSTANT_OUT)
                string(APPEND explicit_Instant_string "${EXPLICIT_INSTANT_OUT}")
                
                if(${thrustarch_it} EQUAL 0 AND ${thrustarch_it2} EQUAL 0)
                  string(APPEND ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_DIMS2_REALS2_MAXDEPTHS2
                         " \\\nELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_DIM2_REAL2_MAXDEPTH2(${dim}, ${real}, ${maxDepth}, ${dim2}, ${real2}, ${maxDepth2})")
                endif(${thrustarch_it} EQUAL 0 AND ${thrustarch_it2} EQUAL 0)
                
              endforeach(maxDepth2)
            endforeach(real2)
          endforeach(dim2)
        endforeach(thrustarch_it2)
      endforeach(maxDepth)
    endforeach(real)
  endforeach(dim)
endforeach(thrustarch_it)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.cu)
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.cu explicit_Instant_file)
else()
  set(explicit_Instant_file "")
endif()

# only overwrite existing source file if changes happened (changes in header files lead to re-compilation anyway)
if(NOT "${explicit_Instant_string}" STREQUAL "${explicit_Instant_file}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.cu
       "${explicit_Instant_string}")
endif()

# creates an object file containing all copy constructors of all possible ImplicitBoxTree<thrustarch,dim> classes
cuda_compile(copy_constructors ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_copy_constructors.cu)
cuda_add_library(copy_constructors_shared ${copy_constructors} SHARED)
if(NOT ${useTBB} EQUAL -1)
  target_link_libraries(copy_constructors_shared tbb)
endif()
list(APPEND libList "copy_constructors_shared")







foreach(thrustarch_it RANGE ${archs_max_index})
  
  list(GET ThrustArchitectures ${thrustarch_it} thrustarch)
  
  foreach(dim ${Dimensions})
    
    foreach(real ${Reals})
      
      foreach(maxDepth_it RANGE ${maxDepths_max_index})
        
        list(GET MaxDepths ${maxDepth_it} maxDepth)
        
        set(object_files_${thrustarch}_${dim}_${real}_${maxDepth} "")
        
        if(${thrustarch_it} EQUAL 0)
          string(APPEND ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS
                 " \\\nELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH(${dim}, ${real}, ${maxDepth})")
        endif(${thrustarch_it} EQUAL 0)
        
        
        # explicitly instantiate  class ImplicitBoxTree<${thrustarch},${dim}>  
        # and all its non-templated member functions
        
        configure_file(src/include/explicit_Instant_THRUSTARCH_DIM_REAL_MAXDEPTH.hpp.in
                       ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_temp.cu @ONLY)
        file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_temp.cu explicit_Instant_string)
        
        if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}.cu)
          file(READ
               ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}.cu
               explicit_Instant_file)
        else()
          set(explicit_Instant_file "")
        endif()
        
        # only overwrite existing source file if changes happened
        # (changes in header files lead to re-compilation anyway)
        if(NOT "${explicit_Instant_string}" STREQUAL "${explicit_Instant_file}")
          file(WRITE
               ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}.cu
               "${explicit_Instant_string}")
        endif()
        
        # creates an object file containing ImplicitBoxTree<thrustarch,dim> and all its non-template members
        cuda_compile(${thrustarch}_${dim}_${real}_${maxDepth}
                     ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}.cu)
        list(APPEND object_files_${thrustarch}_${dim}_${real}_${maxDepth} ${${thrustarch}_${dim}_${real}_${maxDepth}})
        
        
        foreach(grid_it RANGE ${grids_max_index}) #iterate from 0 to nr_grids-1
          
          list(GET Grids ${grid_it} grid)
          list(GET ${grid} 0 n_grid_params)
          
          if(${thrustarch_it} EQUAL 0 AND ${maxDepth_it} EQUAL 0)
            string(APPEND ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS
                   " \\\nELSE_IF_COMBINATION_OF_DIM_REAL_GRID(${dim}, ${real}, ${grid}, ${n_grid_params})")
          endif(${thrustarch_it} EQUAL 0 AND ${maxDepth_it} EQUAL 0)
          
          if(${max_n_params} LESS ${n_grid_params})
            set(max_n_params ${n_grid_params})
          endif()
          
          foreach(map_it RANGE ${maps_max_index}) #iterate from 0 to nr_maps-1
            
            list(GET Maps ${map_it} map)
            list(GET ${map} 0 map_dim) # map_dim == "dim1 dim2 dim3 ..."
            list(GET ${map} 1 map_construct) # map_construct == "con1 con2 con3 ..."
            
            string(REPLACE " " ";" map_dim_list ${map_dim}) # map_dim_list == "dim1;dim2;dim3;..."
            string(REPLACE " " ";" map_construct_list ${map_construct}) # map_construct_list == "con1;con2;con3;..."
            
            list(LENGTH map_dim_list nr_sub_maps)
            math(EXPR sub_map_max_index ${nr_sub_maps}-1)
            
            foreach(sub_map_it RANGE ${sub_map_max_index})
              
              list(GET map_dim_list ${sub_map_it} sub_map_dim)
              list(GET map_construct_list ${sub_map_it} sub_map_construct)
              
              if(${dim} EQUAL ${sub_map_dim})
                
                if(${thrustarch_it} EQUAL 0)
                  if(${maxDepth_it} EQUAL 0 AND ${grid_it} EQUAL 0)
                    string(APPEND ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS
                           " \\\nELSE_IF_COMBINATION_OF_DIM_REAL_MAP(${dim}, ${real}, ${map}, ${sub_map_construct})")
                  endif(${maxDepth_it} EQUAL 0 AND ${grid_it} EQUAL 0)
                  string(APPEND ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS 
                         " \\\nELSE_IF_COMBINATION_OF_DIM_REAL_MAXDEPTH_GRID_MAP(${dim}, ${real}, ${maxDepth}, ${grid}, ${n_grid_params}, ${map}, ${sub_map_construct})")
                endif(${thrustarch_it} EQUAL 0)
          
                if(${max_n_params} LESS ${sub_map_construct})
                  set(max_n_params ${sub_map_construct})
                endif()
                
                
                # explicitly instantiate all member functions of class ImplicitBoxTree<${thrustarch},${dim}> that
                # depend on ${grid} and ${map}
                
                configure_file(src/include/explicit_Instant_THRUSTARCH_DIM_REAL_MAXDEPTH_GRID_MAP.hpp.in
                               ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_temp.cu @ONLY)
                file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_temp.cu explicit_Instant_string)
                
                if(EXISTS
                   ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map}.cu)
                  file(READ
                       ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map}.cu
                       explicit_Instant_file)
                else()
                  set(explicit_Instant_file "")
                endif()
                
                # only overwrite existing source file if changes happened
                # (changes in header files lead to re-compilation anyway)
                if(NOT "${explicit_Instant_string}" STREQUAL "${explicit_Instant_file}")
                  file(WRITE
                       ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map}.cu
                       "${explicit_Instant_string}")
                endif()
                
                # creates an object file containing all template members of ImplicitBoxTree<thrustarch,dim>
                cuda_compile(${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map} 
                             ${CMAKE_CURRENT_BINARY_DIR}/src/include/explicit_Instant_${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map}.cu)
                list(APPEND object_files_${thrustarch}_${dim}_${real}_${maxDepth} ${${thrustarch}_${dim}_${real}_${maxDepth}_${grid}_${map}})
                
              endif(${dim} EQUAL ${sub_map_dim})
              
            endforeach(sub_map_it)
            
          endforeach(map_it)
        endforeach(grid_it)
        
        cuda_add_library(${thrustarch}_${dim}_${real}_${maxDepth}_shared ${object_files_${thrustarch}_${dim}_${real}_${maxDepth}} SHARED)
        if(NOT ${useTBB} EQUAL -1)
          target_link_libraries(${thrustarch}_${dim}_${real}_${maxDepth}_shared tbb)
        endif()
        list(APPEND libList "${thrustarch}_${dim}_${real}_${maxDepth}_shared")
    
      endforeach(maxDepth_it)
    endforeach(real)
  endforeach(dim)
endforeach(thrustarch_it)


add_library(B12_shared INTERFACE)
target_link_libraries(B12_shared INTERFACE ${libList})


install(FILES src/include/BitPattern.h
              src/include/BitPattern.hpp
              src/include/Box.h
              src/include/Box.hpp
              src/include/CooMatrix.h
              src/include/CooMatrix.hpp
              src/include/coordinateFunctions.h
              src/include/coordinateFunctions.hpp
              src/include/ExtendedUInt64_t.h
              src/include/ExtendedUInt64_t.hpp
              src/include/Flags.h
              src/include/Flags.hpp
              src/include/Functors.h
              src/include/Functors.hpp
              src/include/Grids.h
              src/include/Grids.hpp
              src/include/helpFunctions.h
              src/include/helpFunctions.hpp
              src/include/ImplicitBoxTree.h
              src/include/ImplicitBoxTree.hpp
              src/include/Maps.h
              src/include/MapTools.h
              src/include/MapTools.hpp
              src/include/mathFunctions.h
              src/include/mathFunctions.hpp
              src/include/ThrustSystem.h
              src/include/ThrustSystem.hpp
              src/include/TypeDefinitions.h
              src/include/TypeDefinitions.hpp
        DESTINATION ${B12InstallPath}/include)

foreach(map ${Maps})
  install(FILES src/include/Map_${map}.h DESTINATION ${B12InstallPath}/include)
endforeach(map)


install(TARGETS B12_shared ${libList}
        RUNTIME DESTINATION ${B12InstallPath}/bin
        LIBRARY DESTINATION ${B12InstallPath}/lib
        ARCHIVE DESTINATION ${B12InstallPath}/lib/static)


configure_file(B12Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${B12Version}Config.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${B12Version}Config.cmake DESTINATION ${B12InstallPath})


# ------------------------------------ now the Matlab part -------------------------------------


if(MATLAB_FOUND)
  
  cuda_include_directories(${Matlab_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/matlab)
  
  
  # write created preprocessor macros to preprocessor_combinations.h
  foreach(n_params RANGE 2 ${max_n_params} 1)
    math(EXPR temp ${n_params}-1)
    string(APPEND ALL_N_PARAMS
           "\n#define N_PARAMS_${n_params}(param_field) N_PARAMS_${temp}(param_field), param_field[${temp}]")
  endforeach()
  
  set(preprocessor_combinations_string "\n#define FIRST_DIM ${FIRST_DIM} \n
                                        \n#define FIRST_REAL ${FIRST_REAL}\n#define FIRST_REAL_STRING \"${FIRST_REAL}\" \n
                                        \n#define FIRST_MAXDEPTH ${FIRST_MAXDEPTH} \n
                                        \n#define N_PARAMS_0(param_field)\n#define N_PARAMS_1(param_field) N_PARAMS_0(param_field) param_field[0]${ALL_N_PARAMS} \n
                                        \n#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS ${ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS} \n
                                        \n#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_DIMS2_REALS2_MAXDEPTHS2 ${ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_DIMS2_REALS2_MAXDEPTHS2} \n
                                        \n#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS ${ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_GRIDS} \n
                                        \n#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS ${ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAPS} \n
                                        \n#define ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS ${ALL_ELSE_IF_COMBINATIONS_OF_DIMS_REALS_MAXDEPTHS_GRIDS_MAPS} \n")
  
  if(EXISTS ${CMAKE_SOURCE_DIR}/src/matlab/preprocessor_combinations.h)
    file(READ ${CMAKE_SOURCE_DIR}/src/matlab/preprocessor_combinations.h preprocessor_combinations_file)
  else()
    set(preprocessor_combinations_file "")
  endif()
  
  if(NOT "${preprocessor_combinations_string}" STREQUAL "${preprocessor_combinations_file}")
    file(WRITE ${CMAKE_SOURCE_DIR}/src/matlab/preprocessor_combinations.h "${preprocessor_combinations_string}")
  endif()
  
  
  
  # perhaps create ImplicitBoxTree.m
  if(NOT ${useCPP} EQUAL -1)
    set(thrustarch "CPP")
    
    configure_file(src/matlab/general_implicit_box_tree_interface_mex.hpp.in
                  ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex_temp.cu @ONLY)
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex_temp.cu implicit_box_string)
    
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex.cu)
      file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex.cu implicit_box_file)
    else()
      set(implicit_box_file "")
    endif()
    
    # only overwrite existing source file if changes happened (changes in header files lead to re-compilation anyway)
    if(NOT "${implicit_box_string}" STREQUAL "${implicit_box_file}")
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex.cu "${implicit_box_string}")
    endif()
    
    cuda_add_library(implicit_box_tree_interface_mex
                     ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/implicit_box_tree_interface_mex.cu SHARED
                     OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA}
                             -Xcompiler;-fno-var-tracking-assignments)
    set_target_properties(implicit_box_tree_interface_mex PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}")
    target_link_libraries(implicit_box_tree_interface_mex ${Matlab_LIBRARIES} B12_shared)
    
    install(TARGETS implicit_box_tree_interface_mex 
            RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
            LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
            ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
    
    set(arch "CPU")
    set(parallelMethod "cpu")
    set(parallelText "All algorithms work sequentially.")
    set(ImplicitBoxTreeType "ImplicitBoxTree")
    set(implicit_box_tree_interface_mex_func "implicit_box_tree_interface_mex")
    configure_file(src/matlab/generalImplicitBoxTree.m.in ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/ImplicitBoxTree.m @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/ImplicitBoxTree.m DESTINATION ${B12InstallPath}/matlab)
  endif()
  
  
  # perhaps create ompImplicitBoxTree.m
  if(NOT ${useOMP} EQUAL -1)
    set(thrustarch "OMP")
    
    configure_file(src/matlab/general_implicit_box_tree_interface_mex.hpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex_temp.cu @ONLY)
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex_temp.cu implicit_box_string)
    
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex.cu)
      file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex.cu implicit_box_file)
    else()
      set(implicit_box_file "")
    endif()
    
    # only overwrite existing source file if changes happened (changes in header files lead to re-compilation anyway)
    if(NOT "${implicit_box_string}" STREQUAL "${implicit_box_file}")
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex.cu
           "${implicit_box_string}")
    endif()
    
    cuda_add_library(omp_implicit_box_tree_interface_mex
                     ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/omp_implicit_box_tree_interface_mex.cu SHARED
                     OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA}
                             -Xcompiler;-fno-var-tracking-assignments)
    set_target_properties(omp_implicit_box_tree_interface_mex PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}")
    target_link_libraries(omp_implicit_box_tree_interface_mex ${Matlab_LIBRARIES} B12_shared)
    
    install(TARGETS omp_implicit_box_tree_interface_mex 
            RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
            LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
            ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
  
    set(arch "CPU")
    set(parallelMethod "omp")
    set(parallelText "All algorithms are parallelised via OpenMP.")
    set(ImplicitBoxTreeType "ompImplicitBoxTree")
    set(implicit_box_tree_interface_mex_func "omp_implicit_box_tree_interface_mex")
    configure_file(src/matlab/generalImplicitBoxTree.m.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/ompImplicitBoxTree.m @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/ompImplicitBoxTree.m DESTINATION ${B12InstallPath}/matlab)
  endif()
  
  
  # perhaps create tbbImplicitBoxTree.m
  if(NOT ${useTBB} EQUAL -1)
    set(thrustarch "TBB")
    
    configure_file(src/matlab/general_implicit_box_tree_interface_mex.hpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex_temp.cu @ONLY)
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex_temp.cu implicit_box_string)
    
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex.cu)
      file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex.cu implicit_box_file)
    else()
      set(implicit_box_file "")
    endif()
    
    # only overwrite existing source file if changes happened (changes in header files lead to re-compilation anyway)
    if(NOT "${implicit_box_string}" STREQUAL "${implicit_box_file}")
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex.cu
           "${implicit_box_string}")
    endif()
    
    cuda_add_library(tbb_implicit_box_tree_interface_mex
                     ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbb_implicit_box_tree_interface_mex.cu SHARED
                     OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA}
                             -Xcompiler;-fno-var-tracking-assignments)
    set_target_properties(tbb_implicit_box_tree_interface_mex PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}")
    target_link_libraries(tbb_implicit_box_tree_interface_mex ${Matlab_LIBRARIES} B12_shared)
    
    install(TARGETS tbb_implicit_box_tree_interface_mex 
            RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
            LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
            ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
  
    set(arch "CPU")
    set(parallelMethod "tbb")
    set(parallelText "All algorithms are parallelised via OpenMP.")
    set(ImplicitBoxTreeType "tbbImplicitBoxTree")
    set(implicit_box_tree_interface_mex_func "tbb_implicit_box_tree_interface_mex")
    configure_file(src/matlab/generalImplicitBoxTree.m.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbbImplicitBoxTree.m @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/tbbImplicitBoxTree.m DESTINATION ${B12InstallPath}/matlab)
  endif()
  
  
  # perhaps create gpuImplicitBoxTree.m
  if(NOT ${useCUDA} EQUAL "-1")
    set(thrustarch "CUDA")
    
    configure_file(src/matlab/general_implicit_box_tree_interface_mex.hpp.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex_temp.cu @ONLY)
    file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex_temp.cu implicit_box_string)
    
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex.cu)
      file(READ ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex.cu implicit_box_file)
    else()
      set(implicit_box_file "")
    endif()
    
    # only overwrite existing source file if changes happened (changes in header files lead to re-compilation anyway)
    if(NOT "${implicit_box_string}" STREQUAL "${implicit_box_file}")
      file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex.cu
           "${implicit_box_string}")
    endif()
    
    cuda_add_library(gpu_implicit_box_tree_interface_mex
                     ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpu_implicit_box_tree_interface_mex.cu SHARED
                     OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA}
                             -Xcompiler;-fno-var-tracking-assignments)
    set_target_properties(gpu_implicit_box_tree_interface_mex PROPERTIES PREFIX "" SUFFIX ".${Matlab_MEX_EXTENSION}")
    target_link_libraries(gpu_implicit_box_tree_interface_mex ${Matlab_LIBRARIES} B12_shared)
    
    install(TARGETS gpu_implicit_box_tree_interface_mex 
            RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
            LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
            ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
  
    set(arch "GPU")
    set(parallelMethod "gpu")
    set(parallelText "All algorithms are parallelised via CUDA.")
    set(ImplicitBoxTreeType "gpuImplicitBoxTree")
    set(implicit_box_tree_interface_mex_func "gpu_implicit_box_tree_interface_mex")
    configure_file(src/matlab/generalImplicitBoxTree.m.in
                   ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpuImplicitBoxTree.m @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/src/matlab/gpuImplicitBoxTree.m DESTINATION ${B12InstallPath}/matlab)
  endif()
  
  
  # create GridFunctor.m
  cuda_add_library(grid_functor_interface_mex src/matlab/grid_functor_interface_mex.cu SHARED
                   OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA})
  set_target_properties(grid_functor_interface_mex PROPERTIES PREFIX ""
                                                              SUFFIX ".${Matlab_MEX_EXTENSION}"
                                                              COMPILE_FLAGS "-Xcompiler;-fno-var-tracking-assignments")
  target_link_libraries(grid_functor_interface_mex ${Matlab_LIBRARIES})
  
  install(TARGETS grid_functor_interface_mex 
          RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
          LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
          ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
  install(FILES src/matlab/GridFunctor.m DESTINATION ${B12InstallPath}/matlab)
  
  
  # create MapFunctor.m
  cuda_add_library(map_functor_interface_mex src/matlab/map_functor_interface_mex.cu SHARED
                   OPTIONS -DUSE_CPP=${useCPP} -DUSE_OMP=${useOMP} -DUSE_TBB=${useTBB} -DUSE_CUDA=${useCUDA})
  set_target_properties(map_functor_interface_mex PROPERTIES PREFIX ""
                                                             SUFFIX ".${Matlab_MEX_EXTENSION}"
                                                             COMPILE_FLAGS "-Xcompiler;-fno-var-tracking-assignments")
  target_link_libraries(map_functor_interface_mex ${Matlab_LIBRARIES})
  
  install(TARGETS map_functor_interface_mex 
          RUNTIME DESTINATION ${B12InstallPath}/matlab/bin 
          LIBRARY DESTINATION ${B12InstallPath}/matlab/mex 
          ARCHIVE DESTINATION ${B12InstallPath}/matlab/mex/static)
  install(FILES src/matlab/MapFunctor.m DESTINATION ${B12InstallPath}/matlab)
  
  
  install(FILES src/matlab/chainRecurrentSet.m
                src/matlab/continuation.m
                src/matlab/fixedBoxes.m
                src/matlab/readForGaio.m
                src/matlab/subdivision.m
                src/matlab/subdivisionByNim.m
          DESTINATION ${B12InstallPath}/matlab)
  
  
endif(MATLAB_FOUND)




# --------------------- Datei des Nutzers ------------------
# 
# cmake_minimum_required(VERSION 3.5)
# 
# find_package(B12c REQUIRED)
# cuda_include_directories(${B12c_INCLUDE_DIRS})
# cuda_add_executable(yourTarget <yourSources>)
# target_link_libraries(yourTarget ${B12c_LIBRARIES})

