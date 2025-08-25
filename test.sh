#!/bin/bash

BASE_URL="http://localhost:8080"
DB_NAME="ragdb"
TABLE_NAME="documents"

echo "Creating database..."
curl -s -X POST "$BASE_URL/createdb?Name=$DB_NAME"
echo

echo "Creating table..."
curl -s -X POST "$BASE_URL/createtable?Name=$TABLE_NAME&Capacity=100&DB=$DB_NAME"
echo

echo "Inserting 50 diverse rows..."

texts=(
  "The quick brown fox jumps over the lazy dog."
  "Artificial intelligence is transforming technology."
  "Go is a statically typed compiled language designed at Google."
  "Rust offers memory safety without garbage collection."
  "Gin is a web framework written in Go."
  "Machine learning enables computers to learn from data."
  "Natural language processing allows machines to understand text."
  "Neural networks are inspired by the human brain."
  "Cloud computing provides scalable resources on demand."
  "Kubernetes manages containerized applications."
  "Distributed systems are challenging to design and debug."
  "Data science combines statistics and computer science."
  "Blockchain enables decentralized ledgers."
  "Quantum computing uses principles of quantum mechanics."
  "Edge computing processes data near the source."
  "Microservices architecture breaks applications into smaller parts."
  "DevOps bridges development and operations teams."
  "Continuous integration helps maintain code quality."
  "Big data analytics extracts insights from massive datasets."
  "Cybersecurity protects systems from attacks."
  "Augmented reality overlays digital content on the real world."
  "Virtual reality immerses users in a simulated environment."
  "5G networks enable faster wireless communication."
  "Autonomous vehicles rely on sensors and AI."
  "Internet of Things connects everyday devices."
  "Graph databases store data in nodes and edges."
  "Reinforcement learning trains agents through rewards."
  "Generative models create new data from learned patterns."
  "Semantic search improves information retrieval accuracy."
  "Knowledge graphs represent relationships between entities."
  "Data lakes store large amounts of raw data."
  "ETL pipelines extract, transform, and load data."
  "Image recognition is a computer vision task."
  "Speech recognition converts audio to text."
  "Sentiment analysis detects emotions in text."
  "Recommendation systems suggest products or content."
  "Time series analysis studies datasets ordered in time."
  "Anomaly detection identifies unusual patterns."
  "Feature engineering improves machine learning models."
  "Transfer learning uses pretrained models for new tasks."
  "Open source software encourages collaboration."
  "API design affects developer experience."
  "Software testing ensures code correctness."
  "Agile methodology promotes iterative development."
  "Scrum is a framework for managing projects."
  "Kanban visualizes workflow."
  "Pair programming improves code quality."
  "Code reviews catch bugs early."
  "Refactoring cleans up code without changing behavior."
  "Technical debt accumulates from quick fixes."
  "Documentation helps maintain software."
)

urlencode() {
  # URL encode string passed as $1
  local LANG=C
  local length="${#1}"
  local i c
  for (( i = 0; i < length; i++ )); do
    c=${1:i:1}
    case $c in
      [a-zA-Z0-9.~_-]) printf '%s' "$c" ;;
      ' ') printf '%%20' ;;
      *) printf '%%%02X' "'$c"
    esac
  done
}

for i in $(seq 0 49); do
  text="${texts[i]}"
  encoded_text=$(urlencode "$text")
  curl -s -X POST "$BASE_URL/insert?DBName=$DB_NAME&TableName=$TABLE_NAME&Query=$encoded_text" > /dev/null
done

echo "Inserted 50 diverse rows."

echo
echo "Running 5 queries..."

queries=(
    "what is ai",
    "what is llm",
    "finance",
    "reddis"
)

for q in "${queries[@]}"; do
  encoded_q=$(urlencode "$q")
  echo "Query: $q"
  curl -s -X POST "$BASE_URL/query?DBName=$DB_NAME&TableName=$TABLE_NAME&Query=$encoded_q"
  echo
done