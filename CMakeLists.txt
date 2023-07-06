cmake_minimum_required(VERSION 2.8.9)

project(SlicerGiftiLoader)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/mcespedes99/SlicerGiftiLoader")
set(EXTENSION_CATEGORY "Surface Models")
set(EXTENSION_CONTRIBUTORS "Mauricio Cespedes (Western University)")
set(EXTENSION_DESCRIPTION "This extension converts gifti surfaces along with multiple scalars and color tables to VTK format and loads them into 3D Slicer.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/mcespedes99/SlicerGiftiLoader/general/GiftiLoader.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/mcespedes99/SlicerGiftiLoader/general/resources/imgs/SlicerGiftiLoaderScreenshot.png https://raw.githubusercontent.com/mcespedes99/SlicerGiftiLoader/general/resources/imgs/files_selected.png")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(GiftiLoader)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
