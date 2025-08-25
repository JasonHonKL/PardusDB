# Pardus

## Motivation

Pardus is a Redis-like system created specifically for RAG (Retrieval-Augmented Generation) search. Currently, there are many RAG databases. Most of them consistently store data on the hard disk, which is inefficient when handling simple read requests. Here's an example we would like to solve:

User: What is a vision language model?  
Bot → Return a document / string / anything  
User: Tell me something related to vision language models.

At this moment, the query has to be embedded and searched through the whole database again as a vector, which takes O(n) time for a normal full search.

This is inefficient. For example, when we want to retrieve some simple information from an e-commerce website, some very common keys would appear frequently, and certainly not everyone would want to wait so long for this purpose. Hence, we decided to develop Pardus, a Redis-like system designed to solve this problem.

## System Design

Here is the detailed system design.

### Functional Requirements (based on priority)

1. `Query("user query", "table name")`  
   - If exists, return docs/string  
   - If not, fetch from the underlying database  
2. `CreateTable("table name", "sample vector")`

### Non-functional Requirements

- Concurrent reads  
- Timeout  

We use golang for this application

### Roadmap

- Write an HTTP server (for simplicity)
- Support vector search
- Do the caching part 