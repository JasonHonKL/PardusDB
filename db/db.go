package db

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"pardusdb/embed"
	"sort"
	"time"
)

// the key db file is placed here
const MODEL = "nomic-embed-text:latest"
const THRESHOLD = 0.9

type PardusDB struct {
	Name   string
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

func CreateDB(name string) PardusDB {
	return PardusDB{
		Name:   name,
		Tables: map[string]*Table{},
	}
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

type sim struct {
	sim   float32
	index uint32
}

// some mutex shd be done here but let's finish the prototype first
func Query(
	prompt string, table *Table,
) (Val, error) {
	// the function should be embeded the prompt and hen calculate
	if table.Count == 0 {
		// query database
		return Val{}, nil
	}

	vector, err := embed.OllamaEmbedding(prompt, MODEL)

	if err != nil {
		slog.Error("embedding error", "error", err)
		return Val{}, err
	}

	simScore := []sim{}

	for i := range min(table.Capacity, table.Count) {
		s, err := similarity(table.Layers[i].Centriod, vector)
		if err != nil {
			return Val{}, err
		}
		simScore = append(simScore, sim{
			sim:   s,
			index: i,
		})
	}

	sort.Slice(simScore, func(i, j int) bool {
		return simScore[i].sim > simScore[j].sim
	})

	layer := table.Layers[simScore[0].index]

	simScore = []sim{}
	for i, ele := range layer.Data {
		s, _ := similarity(ele.Vector, vector)
		simScore = append(simScore, sim{sim: s, index: uint32(i)})
	}

	sort.Slice(simScore, func(i, j int) bool {
		return simScore[i].sim > simScore[j].sim
	})

	// if value --> insert
	if simScore[0].sim > float32(THRESHOLD) {
		return layer.Data[simScore[0].index].Value, nil
	}
	// return value not found
	return layer.Data[simScore[0].index].Value, nil
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
			newCentroid(layer.Centriod, vector, float32(len(vector)))
	}

	table.pointer = (table.pointer + 1) % table.Capacity
	table.Count += 1

	return nil
}

func newCentroid(
	centroid, point []float32, size float32,
) []float32 {
	n_c := []float32{}

	for i := range int(size) {
		n_c = append(n_c, centroid[i]*size+point[i])
	}
	return n_c
}

func similarity(a, b []float32) (float32, error) {
	size_a := len(a)
	size_b := len(b)

	if size_a != size_b {
		return 0.0, errors.New("different vector size")
	}

	dot_product := float32(0.0)
	norm_a := float32(0.0)
	norm_b := float32(0.0)

	for i := range size_a {
		dot_product += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	}

	if norm_a == float32(0.0) {
		norm_a = float32(.00001)
	}

	if norm_b == float32(0.0) {
		norm_b = float32(.00001)
	}

	return dot_product /
		float32(math.Sqrt(float64(norm_a))) * float32(math.Sqrt(float64(norm_b))), nil
}
