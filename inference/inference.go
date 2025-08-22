package inference

/*
#cgo CFLAGS: -I.
#include "inference.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func FileReader(filename string) *C.FILE {
	cFilename := C.CString(filename)
	defer C.free(unsafe.Pointer(cFilename))
	return C.filereader(cFilename)
}

func CloseFileReader(f *C.FILE) {
	C.closefile(f)
}

func GGUFCheck(f *C.FILE) {
	b := C.magic_word(f)
	fmt.Println(b)
}
