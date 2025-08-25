#!/bin/bash

BASE_URL="http://localhost:8080"
DB_NAME="ragdb"
TABLE_NAME="documents"

urlencode() {
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

queries=(
  "what is ai"
  "what is llm"
  "finance"
  "reddis"
)

echo "Running 5 queries with timing..."

for q in "${queries[@]}"; do
  encoded_q=$(urlencode "$q")
  echo "Query: $q"

  # Use /usr/bin/time to measure curl duration and capture output
  # -p prints simple real/user/sys times
  # Redirect curl output to file, time output to stderr

  RESPONSE_FILE=$(mktemp)
  TIME_OUTPUT=$(mktemp)

  /usr/bin/time -p -o "$TIME_OUTPUT" curl -s -X POST "$BASE_URL/query?DBName=$DB_NAME&TableName=$TABLE_NAME&Query=$encoded_q" > "$RESPONSE_FILE"

  echo "Response:"
  cat "$RESPONSE_FILE"
  echo "Time taken:"
  cat "$TIME_OUTPUT"
  echo "-------------------------------------"

  rm -f "$RESPONSE_FILE" "$TIME_OUTPUT"
done