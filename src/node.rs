use std::sync::Arc;

/// Unique identifier for a node in the graph.
pub type NodeId = u32;

/// A node in the vector graph.
/// Stores the vector data and its neighbors (edges).
#[derive(Clone, Debug)]
pub struct Node<T> {
    /// The vector data
    pub vector: Arc<Vec<T>>,
    /// Indices of neighboring nodes
    pub neighbors: Vec<NodeId>,
    /// Whether this node has been deleted
    pub deleted: bool,
}

impl<T> Node<T> {
    pub fn new(vector: Vec<T>) -> Self {
        Node {
            vector: Arc::new(vector),
            neighbors: Vec::new(),
            deleted: false,
        }
    }

    pub fn with_capacity(vector: Vec<T>, capacity: usize) -> Self {
        Node {
            vector: Arc::new(vector),
            neighbors: Vec::with_capacity(capacity),
            deleted: false,
        }
    }

    /// Add a neighbor if not already present.
    pub fn add_neighbor(&mut self, neighbor_id: NodeId) -> bool {
        if self.neighbors.contains(&neighbor_id) {
            return false;
        }
        self.neighbors.push(neighbor_id);
        true
    }

    /// Remove a neighbor.
    pub fn remove_neighbor(&mut self, neighbor_id: NodeId) -> bool {
        let original_len = self.neighbors.len();
        self.neighbors.retain(|&id| id != neighbor_id);
        self.neighbors.len() != original_len
    }

    /// Mark node as deleted.
    pub fn mark_deleted(&mut self) {
        self.deleted = true;
        self.neighbors.clear();
    }
}

/// A candidate node during search, with its distance to the query.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub id: NodeId,
    pub distance: f32,
}

impl Candidate {
    pub fn new(id: NodeId, distance: f32) -> Self {
        Candidate { id, distance }
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// Order by distance (ascending - closer is "smaller")
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node: Node<f32> = Node::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(node.vector.len(), 3);
        assert!(node.neighbors.is_empty());
        assert!(!node.deleted);
    }

    #[test]
    fn test_add_neighbor() {
        let mut node: Node<f32> = Node::new(vec![1.0, 2.0]);
        assert!(node.add_neighbor(1));
        assert!(node.add_neighbor(2));
        assert!(!node.add_neighbor(1)); // Already exists
        assert_eq!(node.neighbors.len(), 2);
    }

    #[test]
    fn test_remove_neighbor() {
        let mut node: Node<f32> = Node::new(vec![1.0, 2.0]);
        node.add_neighbor(1);
        node.add_neighbor(2);
        assert!(node.remove_neighbor(1));
        assert!(!node.remove_neighbor(1)); // Already removed
        assert_eq!(node.neighbors.len(), 1);
    }
}
