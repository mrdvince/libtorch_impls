# https://github.com/prabhuomkar/pytorch-cpp/blob/master/cmake/fetch_cifar10.cmake
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

function(check_files BASE_DIR FILES_TO_CHECK FILE_MD5S MISSING_FILES)
    foreach(FILE_TO_CHECK ${${FILES_TO_CHECK}})
        if(EXISTS "${BASE_DIR}/${FILE_TO_CHECK}")
            list(FIND ${FILES_TO_CHECK} ${FILE_TO_CHECK} MD5_IDX)
            list(GET ${FILE_MD5S} ${MD5_IDX} EXPECTED_FILE_MD5)
            
            file(MD5 "${BASE_DIR}/${FILE_TO_CHECK}" ACTUAL_FILE_MD5)
            
            if(NOT (EXPECTED_FILE_MD5 STREQUAL ACTUAL_FILE_MD5))
                list(APPEND RESULT_FILES ${FILE_TO_CHECK})
            endif()
        else()
            list(APPEND RESULT_FILES ${FILE_TO_CHECK})
        endif()
    endforeach()
    
    set(${MISSING_FILES} ${RESULT_FILES} PARENT_SCOPE)
endfunction()

function(fetch_cifar10 DATA_DIR)
    set(CIFAR_DIR "${DATA_DIR}/cifar10")
    set(CIFAR_URL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
    set(CIFAR_EXTRACTED_FILES
        "data_batch_1.bin" "data_batch_2.bin" "data_batch_3.bin"
        "data_batch_4.bin" "data_batch_5.bin" "test_batch.bin")

    set(CIFAR_EXTRACTED_FILE_MD5s
        "5dd7e06a14cb22eb9f671a540d1b7c25" "5ea93a67294ea407fff1d09f752e9692"
        "942cd6a4bcdd0dd3c604fbe906cb4421" "ae636b3ba5c66a11e91e8cb52e771fcb"
        "53f37980c15c3d472c316c40844f3f0d" "803d5f7f4d78ea53de84dbe85f74fb6d")


    check_files(${CIFAR_DIR} CIFAR_EXTRACTED_FILES CIFAR_EXTRACTED_FILE_MD5s
                MISSING_FILES)

    if(MISSING_FILES)
        message(STATUS "Fetching CIFAR10 dataset...")
        set(CIFAR10_ARCHIVE_NAME "cifar-10-binary.tar.gz")

        file(REMOVE_RECURSE ${CIFAR_DIR})

        file(
            DOWNLOAD ${CIFAR_URL} "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}"
            SHOW_PROGRESS
            EXPECTED_MD5 "c32a1d4ab5d03f1284b67883e8d87530")

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar x
                    "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}"
            WORKING_DIRECTORY ${DATA_DIR})

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E rename
                    "${DATA_DIR}/cifar-10-batches-bin" ${CIFAR_DIR})

        file(REMOVE "${DATA_DIR}/${CIFAR10_ARCHIVE_NAME}")
        message(STATUS "Fetching CIFAR10 dataset - done")
    endif()
endfunction()
