use std::collections::{BinaryHeap, HashSet};

use crate::distance::{Distance, Numeric};
use crate::node::{Candidate, Node, NodeId};

/// Configuration for the graph.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Maximum number of neighbors per node
    pub max_neighbors: usize,
    /// Alpha for strict pruning (typically 1.0)
    pub alpha_strict: f32,
    /// Alpha for relaxed pruning / highways (typically 1.2)
    pub alpha_relaxed: f32,
    /// Buffer size for candidate search
    pub search_buffer: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            max_neighbors: 16,
            alpha_strict: 1.0,
            alpha_relaxed: 1.2,
            search_buffer: 64,  // Reduced from 200 - enough for good recall
        }
    }
}

/// The vector graph structure.
pub struct Graph<T, D>
where
    T: Numeric,
    D: Distance<T>,
{
    /// All nodes in the graph
    nodes: Vec<Node<T>>,
    /// Centroid vector (running average)
    centroid: Vec<f32>,
    /// Number of active (non-deleted) nodes
    active_count: usize,
    /// Free list for deleted node slots
    free_list: Vec<NodeId>,
    /// Configuration
    config: GraphConfig,
    /// Distance metric (zero-sized marker type)
    _metric: std::marker::PhantomData<D>,
}

impl<T, D> Graph<T, D>
where
    T: Numeric,
    D: Distance<T>,
{
    pub fn new(dimension: usize, config: GraphConfig) -> Self {
        Graph {
            nodes: Vec::new(),
            centroid: vec![0.0; dimension],
            active_count: 0,
            free_list: Vec::new(),
            config,
            _metric: std::marker::PhantomData,
        }
    }

    /// Get the dimension of vectors in this graph.
    pub fn dimension(&self) -> usize {
        self.centroid.len()
    }

    /// Get the number of active nodes.
    pub fn len(&self) -> usize {
        self.active_count
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.active_count == 0
    }

    /// Get a node by ID.
    pub fn get(&self, id: NodeId) -> Option<&Node<T>> {
        self.nodes.get(id as usize).filter(|n| !n.deleted)
    }

    /// Get a mutable reference to a node by ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id as usize).filter(|n| !n.deleted)
    }

    /// Get the centroid vector.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Compute distance between two vectors.
    #[inline]
    fn distance(a: &[T], b: &[T]) -> f32 {
        D::compute(a, b)
    }

    /// Compute distance from a node to a vector.
    #[inline]
    fn distance_to_vector(node: &Node<T>, vector: &[T]) -> f32 {
        Self::distance(&node.vector, vector)
    }

    /// Update centroid after inserting a new node.
    /// Uses overflow-safe formula: centroid * n/(n+1) + new_vector/(n+1)
    fn update_centroid_insert(&mut self, vector: &[T]) {
        let n = self.active_count as f32;
        let n_plus_1 = n + 1.0;

        self.centroid
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(c, &v)| {
                *c = *c * n / n_plus_1 + v.to_f32() / n_plus_1;
            });
    }

    /// Update centroid after deleting a node.
    /// Uses overflow-safe formula: centroid * n/(n-1) - deleted_vector/(n-1)
    fn update_centroid_delete(&mut self, vector: &[T]) {
        if self.active_count <= 1 {
            // Reset centroid to zero if last node
            self.centroid.fill(0.0);
            return;
        }

        let n = self.active_count as f32;
        let n_minus_1 = n - 1.0;

        self.centroid
            .iter_mut()
            .zip(vector.iter())
            .for_each(|(c, &v)| {
                *c = *c * n / n_minus_1 - v.to_f32() / n_minus_1;
            });
    }

    /// Find the best starting node for search (first active node).
    fn find_start_node(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .find(|(_, n)| !n.deleted)
            .map(|(id, _)| id as NodeId)
    }

    /// Greedy search from centroid to find candidates close to target.
    /// Returns candidates sorted by distance.
    pub fn search(&self, target: &[T], ef_search: usize) -> Vec<Candidate> {
        if self.is_empty() {
            return Vec::new();
        }

        // Find starting point
        let start = match self.find_start_node() {
            Some(id) => id,
            None => return Vec::new(),
        };

        let start_node = &self.nodes[start as usize];
        let start_dist = Self::distance_to_vector(start_node, target);

        // Min-heap for candidates (BinaryHeap is max-heap, so we reverse the ordering)
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        candidates.push(Candidate::new(start, start_dist));

        // Set of visited nodes - use a Vec for small graphs, HashSet for larger
        let mut visited = vec![false; self.nodes.len()];
        visited[start as usize] = true;

        // Result buffer - use Vec and sort at the end
        let mut results: Vec<Candidate> = Vec::with_capacity(ef_search);
        results.push(Candidate::new(start, start_dist));

        // Track worst distance in results for early termination
        let mut worst_dist = start_dist;

        while let Some(current) = candidates.pop() {
            // Reverse because BinaryHeap is max-heap
            let current = Candidate::new(current.id, -current.distance);
            let current_dist = current.distance;

            // Early termination: if current is further than worst, stop
            if results.len() >= ef_search && current_dist > worst_dist {
                break;
            }

            // Explore neighbors
            if let Some(node) = self.get(current.id) {
                for &neighbor_id in &node.neighbors {
                    let nid = neighbor_id as usize;
                    if nid >= visited.len() || visited[nid] {
                        continue;
                    }
                    visited[nid] = true;

                    if let Some(neighbor) = self.get(neighbor_id) {
                        let dist = Self::distance_to_vector(neighbor, target);

                        // Add to candidates (negate for min-heap behavior)
                        candidates.push(Candidate::new(neighbor_id, -dist));

                        // Add to results if room or better than worst
                        if results.len() < ef_search {
                            results.push(Candidate::new(neighbor_id, dist));
                            if dist > worst_dist {
                                worst_dist = dist;
                            }
                        } else if dist < worst_dist {
                            // Replace worst
                            if let Some(pos) = results.iter().position(|c| c.distance == worst_dist) {
                                results[pos] = Candidate::new(neighbor_id, dist);
                            }
                            // Recompute worst
                            worst_dist = results.iter().map(|c| c.distance).fold(f32::NEG_INFINITY, f32::max);
                        }
                    }
                }
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    /// Robust prune: select diverse neighbors from candidates.
    /// Uses geometric diversity to avoid redundant edges.
    pub fn robust_prune(
        &self,
        target_vector: &[T],
        candidates: &[Candidate],
        alpha: f32,
        max_neighbors: usize,
    ) -> Vec<NodeId> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut selected: Vec<NodeId> = Vec::with_capacity(max_neighbors);

        for candidate in candidates {
            if selected.len() >= max_neighbors {
                break;
            }

            let candidate_node = match self.get(candidate.id) {
                Some(n) => n,
                None => continue,
            };

            // Check if candidate is redundant with already selected neighbors
            let is_redundant = selected.iter().any(|&selected_id| {
                self.get(selected_id).map_or(false, |selected_node| {
                    // Rule: prune if dist(target, candidate) >= (1/alpha) * dist(selected, candidate)
                    let dist_to_target = Self::distance(target_vector, &candidate_node.vector);
                    let dist_to_selected = Self::distance(&selected_node.vector, &candidate_node.vector);

                    dist_to_target >= (1.0 / alpha) * dist_to_selected
                })
            });

            if !is_redundant {
                selected.push(candidate.id);
            }
        }

        selected
    }

    /// Insert a new vector into the graph.
    /// Returns the ID of the newly inserted node.
    pub fn insert(&mut self, vector: Vec<T>) -> NodeId {
        let max_neighbors = self.config.max_neighbors;
        let alpha = self.config.alpha_strict;
        let search_buffer = self.config.search_buffer;

        // Allocate node ID
        let node_id = self.allocate_node_id();

        // Create new node
        let mut new_node = Node::with_capacity(vector.clone(), max_neighbors);

        // Update centroid
        self.update_centroid_insert(&vector);

        // If this is the first node, just add it
        if self.active_count == 1 {
            self.insert_node_at(node_id, new_node);
            return node_id;
        }

        // Search for candidates
        let candidates = self.search(&vector, search_buffer);

        // Prune candidates to get neighbors
        let neighbors = self.robust_prune(&vector, &candidates, alpha, max_neighbors);

        // Set neighbors for new node
        new_node.neighbors = neighbors.clone();
        self.insert_node_at(node_id, new_node);

        // Back-link: add new node to neighbors' neighbor lists
        // Collect neighbors that need reverse pruning
        let mut to_prune: Vec<NodeId> = Vec::new();

        for &neighbor_id in &neighbors {
            if let Some(neighbor) = self.get_mut(neighbor_id) {
                neighbor.add_neighbor(node_id);

                if neighbor.neighbors.len() > max_neighbors {
                    to_prune.push(neighbor_id);
                }
            }
        }

        // Handle reverse pruning for neighbors that exceeded max_neighbors
        for neighbor_id in to_prune {
            self.reverse_prune(neighbor_id, max_neighbors, alpha);
        }

        node_id
    }

    /// Reverse prune a node that may have too many neighbors.
    fn reverse_prune(&mut self, node_id: NodeId, max_neighbors: usize, alpha: f32) {
        let neighbor_ids: Vec<NodeId> = match self.get(node_id) {
            Some(n) => n.neighbors.clone(),
            None => return,
        };

        if neighbor_ids.len() <= max_neighbors {
            return;
        }

        // Compute distances to all neighbors
        let candidates: Vec<Candidate> = {
            let node = match self.get(node_id) {
                Some(n) => n,
                None => return,
            };

            neighbor_ids
                .iter()
                .filter_map(|&nid| {
                    self.get(nid).map(|n| {
                        let dist = Self::distance(&node.vector, &n.vector);
                        Candidate::new(nid, dist)
                    })
                })
                .collect()
        };

        // Re-prune using the node's vector
        let node_vector: Vec<T> = self.get(node_id).unwrap().vector.as_ref().clone();
        let new_neighbors = self.robust_prune(&node_vector, &candidates, alpha, max_neighbors);

        // Update neighbor list
        if let Some(node) = self.get_mut(node_id) {
            node.neighbors = new_neighbors;
        }
    }

    /// Allocate a node ID (reuse from free list or create new).
    fn allocate_node_id(&mut self) -> NodeId {
        self.active_count += 1;

        self.free_list.pop().unwrap_or_else(|| {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Node::new(Vec::new())); // placeholder
            id
        })
    }

    /// Insert a node at a specific ID.
    fn insert_node_at(&mut self, id: NodeId, node: Node<T>) {
        if id as usize >= self.nodes.len() {
            self.nodes
                .resize_with((id + 1) as usize, || Node::new(Vec::new()));
        }
        self.nodes[id as usize] = node;
    }

    /// Delete a node by ID.
    /// Returns true if successful.
    pub fn delete(&mut self, id: NodeId) -> bool {
        let vector = match self.get(id) {
            Some(n) => n.vector.as_ref().clone(),
            None => return false,
        };

        // Get neighbors before deletion
        let neighbors: Vec<NodeId> = self
            .get(id)
            .map(|n| n.neighbors.clone())
            .unwrap_or_default();

        // Remove this node from neighbors' lists
        for &neighbor_id in &neighbors {
            if let Some(neighbor) = self.get_mut(neighbor_id) {
                neighbor.remove_neighbor(id);
            }
        }

        // Update centroid
        self.update_centroid_delete(&vector);

        // Mark node as deleted
        if let Some(node) = self.nodes.get_mut(id as usize) {
            node.mark_deleted();
        }

        // Add to free list
        self.free_list.push(id);
        self.active_count -= 1;

        true
    }

    /// Update a node's vector (delete + reinsert).
    /// Returns true if successful.
    pub fn update(&mut self, id: NodeId, new_vector: Vec<T>) -> bool {
        if self.get(id).is_none() {
            return false;
        }

        self.delete(id);
        self.insert(new_vector);
        true
    }

    /// Query for k nearest neighbors.
    pub fn query(&self, vector: &[T], k: usize, ef_search: usize) -> Vec<Candidate> {
        let ef = ef_search.max(k);
        let mut results = self.search(vector, ef);
        results.truncate(k);
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Euclidean;

    #[test]
    fn test_graph_creation() {
        let graph: Graph<f32, Euclidean> = Graph::new(3, GraphConfig::default());
        assert!(graph.is_empty());
        assert_eq!(graph.dimension(), 3);
    }

    #[test]
    fn test_insert_first_node() {
        let mut graph: Graph<f32, Euclidean> = Graph::new(2, GraphConfig::default());
        let id = graph.insert(vec![1.0, 2.0]);
        assert_eq!(id, 0);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_insert_multiple_nodes() {
        let mut graph: Graph<f32, Euclidean> = Graph::new(2, GraphConfig::default());

        graph.insert(vec![0.0, 0.0]);
        graph.insert(vec![1.0, 1.0]);
        graph.insert(vec![2.0, 2.0]);

        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_query() {
        let mut graph: Graph<f32, Euclidean> = Graph::new(2, GraphConfig::default());

        graph.insert(vec![0.0, 0.0]);
        graph.insert(vec![10.0, 10.0]);
        graph.insert(vec![1.0, 1.0]);

        let results = graph.query(&[0.5, 0.5], 2, 10);

        assert_eq!(results.len(), 2);
        // Results should be ordered by distance (<= because distances can be equal)
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn test_delete() {
        let mut graph: Graph<f32, Euclidean> = Graph::new(2, GraphConfig::default());

        let id = graph.insert(vec![1.0, 2.0]);
        assert_eq!(graph.len(), 1);

        assert!(graph.delete(id));
        assert_eq!(graph.len(), 0);
    }
}
