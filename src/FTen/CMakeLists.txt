include_directories(generic)

set(FTen_Gen_CPPS
    generic/FArray.cpp
)

add_library(FTen_LIB SHARED ${FTen_Gen_CPPS})

install(FILES
    FTensor.h
    DESTINATION "${NNF_INSTUDE_DIR}/inclube")

install(TARGETS FTen_LIB
    LIBRARY DESTINATION "${NNF_INSTUDE_DIR}/generic")