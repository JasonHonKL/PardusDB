package main

import (
	"fmt"
	"net/http"
	"pardusdb/db"
	"pardusdb/net"

	"github.com/gin-gonic/gin"
)

type QueryParam struct {
	Query     string
	TableName string
	DBName    string
}

type CreateDBParam struct {
	Name string
}

type CreateTableParam struct {
	Name     string
	Capacity uint32
	DB       string // db name
}

type InsertParam struct {
	DBName    string
	TableName string
	Query     string // val Text
	Val       db.Val
}

func main() {

	cache := net.Cache{
		Room: map[string]*db.PardusDB{},
	}

	router := gin.Default()

	router.POST("/query", func(ctx *gin.Context) {
		var param QueryParam
		if ctx.ShouldBindQuery(&param) == nil {
			database, found := cache.Room[param.DBName]
			if !found {
				fmt.Println("Database not found")
				return
			}
			table, found := database.Tables[param.TableName]
			if !found {
				fmt.Println("Table not found")
				return
			}
			fmt.Println("get table")
			val, _ := db.Query(param.Query, table)
			ctx.JSON(http.StatusOK, gin.H{
				"val": val.Text,
			})
		}
	})

	router.POST("/insert", func(ctx *gin.Context) {
		var param InsertParam
		if ctx.ShouldBindQuery(&param) == nil {
			// here it actually should fetch the db at the back and save in RAM
			// Now it is just a temp solution
			database, found := cache.Room[param.DBName]
			if !found {
				return
			}
			// TODO: more robust way to handle db.val
			db.InsertRow(param.TableName, param.Query, db.Val{Text: param.Query}, database)
			return
		}
	})

	router.POST("/createdb", func(ctx *gin.Context) {
		var param CreateDBParam
		if ctx.ShouldBindQuery(&param) == nil {
			_, found := cache.Room[param.Name]
			if found {
				ctx.JSON(http.StatusConflict, gin.H{
					"error": "db exists",
				})
				return
			}

			new_db := db.CreateDB(param.Name)
			cache.Room[param.Name] = &new_db

			ctx.JSON(http.StatusOK, gin.H{
				"message": "create successfully",
			})
			return
		}
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "bad request",
		})
	})

	router.POST("/createtable", func(ctx *gin.Context) {
		var param CreateTableParam
		if ctx.ShouldBindQuery(&param) == nil {
			database, found := cache.Room[param.DB]

			if !found {
				ctx.JSON(http.StatusBadRequest, gin.H{
					"error": "database not found",
				})
				return
			}

			db.CreateTable(param.Name, param.Capacity, database)

			ctx.JSON(http.StatusOK, gin.H{
				"message": "create table successfully",
			})
			return
		}
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "bad request",
		})
	})

	router.Run()
}
