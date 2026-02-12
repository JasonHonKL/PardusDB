use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::distance::{Distance, Numeric};
use crate::error::{MarsError, Result};
use crate::graph::{Graph, GraphConfig};
use crate::node::{Candidate, NodeId};
use crate::storage::Storage;

/// Configuration for the vector database.
#[derive(Clone, Debug)]
pub struct Config {
    /// Vector dimension
    pub dimension: usize,
    /// Graph configuration
    pub graph: GraphConfig,
    /// Path to database file (None for in-memory)
    pub path: Option<PathBuf>,
}

impl Config {
    pub fn new(dimension: usize) -> Self {
        Config {
            dimension,
            graph: GraphConfig::default(),
            path: None,
        }
    }

    pub fn with_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_max_neighbors(mut self, max_neighbors: usize) -> Self {
        self.graph.max_neighbors = max_neighbors;
        self
    }

    pub fn with_alpha(mut self, strict: f32, relaxed: f32) -> Self {
        self.graph.alpha_strict = strict;
        self.graph.alpha_relaxed = relaxed;
        self
    }

    pub fn with_search_buffer(mut self, buffer: usize) -> Self {
        self.graph.search_buffer = buffer;
        self
    }
}

/// Search result containing the node ID and distance.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: NodeId,
    pub distance: f32,
}

impl From<Candidate> for SearchResult {
    fn from(c: Candidate) -> Self {
        SearchResult {
            id: c.id,
            distance: c.distance,
        }
    }
}

/// The main vector database interface.
pub struct VectorDB<T, D>
where
    T: Numeric,
    D: Distance<T>,
{
    graph: Arc<RwLock<Graph<T, D>>>,
    storage: Option<Arc<RwLock<Storage>>>,
    config: Config,
}

impl<T, D> VectorDB<T, D>
where
    T: Numeric,
    D: Distance<T>,
{
    /// Create a new in-memory vector database.
    pub fn in_memory(dimension: usize) -> Self {
        VectorDB {
            graph: Arc::new(RwLock::new(Graph::new(dimension, GraphConfig::default()))),
            storage: None,
            config: Config::new(dimension),
        }
    }

    /// Create a new vector database with custom configuration.
    pub fn with_config(config: Config) -> Self {
        VectorDB {
            graph: Arc::new(RwLock::new(Graph::new(
                config.dimension,
                config.graph.clone(),
            ))),
            storage: None,
            config,
        }
    }

    /// Open or create a persistent vector database.
    pub fn open<P: AsRef<Path>>(path: P, dimension: usize) -> Result<Self> {
        let path = path.as_ref();

        let storage = if path.exists() {
            Storage::open(path)?
        } else {
            Storage::create(path, dimension as u32)?
        };

        let config = Config::new(dimension).with_path(path);

        Ok(VectorDB {
            graph: Arc::new(RwLock::new(Graph::new(dimension, config.graph.clone()))),
            storage: Some(Arc::new(RwLock::new(storage))),
            config,
        })
    }

    /// Insert a vector into the database.
    /// Returns the ID of the inserted node.
    pub fn insert(&self, vector: Vec<T>) -> Result<NodeId> {
        let mut graph = self.graph.write().unwrap();

        if vector.len() != self.config.dimension {
            return Err(MarsError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        Ok(graph.insert(vector))
    }

    /// Insert multiple vectors in batch.
    /// Returns the IDs of the inserted nodes.
    pub fn insert_batch(&self, vectors: Vec<Vec<T>>) -> Result<Vec<NodeId>> {
        let mut graph = self.graph.write().unwrap();
        let mut ids = Vec::with_capacity(vectors.len());

        for vector in vectors {
            if vector.len() != self.config.dimension {
                return Err(MarsError::DimensionMismatch {
                    expected: self.config.dimension,
                    actual: vector.len(),
                });
            }
            ids.push(graph.insert(vector));
        }

        Ok(ids)
    }

    /// Query for k nearest neighbors.
    pub fn query(&self, vector: &[T], k: usize) -> Result<Vec<SearchResult>> {
        if vector.len() != self.config.dimension {
            return Err(MarsError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let graph = self.graph.read().unwrap();

        if graph.is_empty() {
            return Ok(Vec::new());
        }

        let ef_search = self.config.graph.search_buffer.max(k);
        let candidates = graph.query(vector, k, ef_search);

        Ok(candidates.into_iter().map(SearchResult::from).collect())
    }

    /// Query with custom ef_search parameter.
    pub fn query_with_ef(&self, vector: &[T], k: usize, ef_search: usize) -> Result<Vec<SearchResult>> {
        if vector.len() != self.config.dimension {
            return Err(MarsError::DimensionMismatch {
                expected: self.config.dimension,
                actual: vector.len(),
            });
        }

        let graph = self.graph.read().unwrap();

        if graph.is_empty() {
            return Ok(Vec::new());
        }

        let candidates = graph.query(vector, k, ef_search);

        Ok(candidates.into_iter().map(SearchResult::from).collect())
    }

    /// Delete a node by ID.
    pub fn delete(&self, id: NodeId) -> Result<bool> {
        let mut graph = self.graph.write().unwrap();
        Ok(graph.delete(id))
    }

    /// Update a node's vector.
    pub fn update(&self, id: NodeId, new_vector: Vec<T>) -> Result<bool> {
        if new_vector.len() != self.config.dimension {
            return Err(MarsError::DimensionMismatch {
                expected: self.config.dimension,
                actual: new_vector.len(),
            });
        }

        let mut graph = self.graph.write().unwrap();
        Ok(graph.update(id, new_vector))
    }

    /// Get a node's vector by ID.
    pub fn get(&self, id: NodeId) -> Option<Vec<T>>
    where
        T: Clone,
    {
        let graph = self.graph.read().unwrap();
        graph.get(id).map(|n| n.vector.as_ref().clone())
    }

    /// Get the number of vectors in the database.
    pub fn len(&self) -> usize {
        self.graph.read().unwrap().len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.read().unwrap().is_empty()
    }

    /// Get the centroid vector.
    pub fn centroid(&self) -> Vec<f32> {
        self.graph.read().unwrap().centroid().to_vec()
    }

    /// Sync to disk (if persistent).
    pub fn sync(&self) -> Result<()> {
        if let Some(storage) = &self.storage {
            let storage = storage.read().unwrap();
            storage.sync()?;
        }
        Ok(())
    }

    /// Get the dimension of vectors.
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
}

/// Type aliases for common configurations
pub type CosineDB<T> = VectorDB<T, crate::distance::Cosine>;
pub type DotProductDB<T> = VectorDB<T, crate::distance::DotProduct>;
pub type EuclideanDB<T> = VectorDB<T, crate::distance::Euclidean>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Euclidean;

    #[test]
    fn test_insert_and_query() {
        let db: EuclideanDB<f32> = VectorDB::in_memory(2);

        db.insert(vec![0.0, 0.0]).unwrap();
        db.insert(vec![1.0, 1.0]).unwrap();
        db.insert(vec![10.0, 10.0]).unwrap();

        let results = db.query(&[0.5, 0.5], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete() {
        let db: EuclideanDB<f32> = VectorDB::in_memory(2);

        let id = db.insert(vec![1.0, 2.0]).unwrap();
        assert_eq!(db.len(), 1);

        assert!(db.delete(id).unwrap());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_update() {
        let db: EuclideanDB<f32> = VectorDB::in_memory(2);

        let id = db.insert(vec![1.0, 2.0]).unwrap();
        db.update(id, vec![3.0, 4.0]).unwrap();

        let vector = db.get(id).unwrap();
        assert_eq!(vector, vec![3.0, 4.0]);
    }

    #[test]
    fn test_dimension_mismatch() {
        let db: EuclideanDB<f32> = VectorDB::in_memory(2);

        let result = db.insert(vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_batch() {
        let db: EuclideanDB<f32> = VectorDB::in_memory(2);

        let ids = db.insert_batch(vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ]).unwrap();

        assert_eq!(ids.len(), 3);
        assert_eq!(db.len(), 3);
    }
}
