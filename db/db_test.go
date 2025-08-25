package db_test

import (
	"fmt"
	"pardusdb/db"
	"testing"
)

func TestCreateTable(t *testing.T) {
	pardus := db.PardusDB{
		Tables: map[string]*db.Table{},
	}

	table, err := db.CreateTable("testing table", 5, &pardus)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(table)

	db.InsertRow("testing table", db.Val{Text: "llama ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "mistral ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "what the hack ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)

	fmt.Println(table)
}
