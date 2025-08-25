package main

import (
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
	Query     string
}

func main() {

	cache := net.Cache{
		Room: map[string]*db.PardusDB{},
	}

	router := gin.Default()

	router.POST("/query", func(ctx *gin.Context) {})

	router.POST("/insert", func(ctx *gin.Context) {

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
		}
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "bad request",
		})
	})

	router.POST("/createtable", func(ctx *gin.Context) {
		var param CreateTableParam
		if ctx.ShouldBindQuery(&param) == nil {
			database, found := cache.Room[param.Name]

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
		}
		ctx.JSON(http.StatusBadRequest, gin.H{
			"error": "bad request",
		})
	})

	router.Run()
}
