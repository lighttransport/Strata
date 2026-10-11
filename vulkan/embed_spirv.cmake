file(READ "${INPUT}" _bytes HEX)
string(REGEX REPLACE "(..)" "0x\\1," _bytes "${_bytes}")
file(WRITE "${OUTPUT}" "#pragma once\nstatic const unsigned char ${NAME}_spv[] = {${_bytes}};\n")
