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

	db.InsertRow("testing table", db.Val{Text: "Llamas are the largest lamoid or South American Camelid species. Unlike Old World Camelids, they do not have humps. "}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "mistral ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "An artificial intelligence (AI) agent is a system that autonomously performs tasks by designing workflows with available tools. AI agents can encompass a wide range of functions beyond natural"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "what the hack ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "who are you ?"}, &pardus)
	db.InsertRow("testing table", db.Val{Text: "hello who are you ?"}, &pardus)

	val, err := db.Query("what is ai agent ?", table)

	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(val.Text)
}
