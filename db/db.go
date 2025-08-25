package db

import (
	"errors"
	"fmt"
	"log/slog"
	"pardusdb/embed"
	"time"
)

// the key db file is placed here
const MODEL = "nomic-embed-text:latest"

type PardusDB struct {
	Tables map[string]*Table
}

type Table struct {
	Name string

	Capacity uint32 //max no. of layer
	Count    uint32 //current layer

	pointer uint32 // round rubin
	Layers  []Layer

	Model string
}

type Layer struct {
	Data     []Object
	Centriod []float32
}

type Object struct {
	Value Val

	Time time.Time

	Vector []float32
}

type Val struct {
	Text     string // this is the comparsion part
	MetaData map[string]any
}

func CreateTable(
	name string, cap uint32, db *PardusDB,
) (*Table, error) {

	_, found := (*db).Tables[name]
	if found {
		return nil, fmt.Errorf("%s already exists in the db", name)
	}

	layers := []Layer{}

	for range cap {
		layers = append(layers, Layer{
			Data:     []Object{},
			Centriod: []float32{},
		})
	}

	t := &Table{
		Name:     name,
		Capacity: cap,
		Count:    0,
		pointer:  0,
		Layers:   layers,
	}
	db.Tables[t.Name] = t

	return t, nil
}

// some mutex shd be done here but let's finish the prototype first
func Query(prompt string, table string) {
	// the function should be embeded the prompt and hen calculate

	vector, err := embed.OllamaEmbedding(prompt, MODEL)

	if err != nil {
		slog.Error("embedding error", "error", err)
		return
	}

	fmt.Println(vector)

	// the difference between the prompt vector and the centroid
	// then follow up with full search from that table

	// if value --> insert
}

func InsertRow(
	name string, val Val, db *PardusDB,
) error {
	// the insert shall be insert in to the nearest centroid one
	vector, err := embed.OllamaEmbedding(val.Text, MODEL)

	if err != nil {
		slog.Error(err.Error())
		return err
	}

	table, found := db.Tables[name]
	if !found {
		return (errors.New("table not found"))
	}

	layer := &table.Layers[table.pointer]

	obj := Object{
		Value:  val,
		Time:   time.Now(),
		Vector: vector,
	}

	layer.Data = append(layer.Data, obj)

	if len(layer.Centriod) == 0 {
		layer.Centriod = vector
	} else {
		layer.Centriod =
			new_centroid(layer.Centriod, vector, float32(len(layer.Data)))
	}

	table.pointer = (table.pointer + 1) % table.Capacity
	fmt.Println(table.pointer)

	return nil
}

func new_centroid(
	centroid, point []float32, size float32,
) []float32 {
	n_c := []float32{}

	for i := range int(size) {
		n_c = append(n_c, centroid[i]*size+point[i])
	}
	return n_c
}

func dot_product(a, b []float32)
