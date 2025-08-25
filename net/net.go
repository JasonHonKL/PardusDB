package net

import "pardusdb/db"

// this net package is for the purpose of hosting the db as a server

type Cache struct {
	Room map[string]*db.PardusDB // each room contain a db
}

// query

// create table
