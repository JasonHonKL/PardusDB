package main

import (
	"fmt"

	"pardus/inference"
)

import "C"

func main() {
	file := inference.FileReader("model.gguf")
	if file == nil {
		fmt.Println("Failed to open file")
		panic("error can't open file")
	} else {
		fmt.Println("File opened successfully")
	}

	inference.CloseFileReader(file)
	fmt.Println("Close file successfully")

}
