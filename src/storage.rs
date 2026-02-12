use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::Path;

use memmap2::MmapMut;

use crate::error::{MarsError, Result};

/// Magic bytes for file format identification
const MAGIC: &[u8; 4] = b"MARS";

/// Current file format version
const VERSION: u32 = 1;

/// File header structure
#[derive(Clone, Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub dimension: u32,
    pub node_count: u32,
    pub centroid_offset: u64,
    pub index_offset: u64,
    pub data_offset: u64,
    pub free_list_head: u32,
}

impl Header {
    pub const SIZE: usize = 4 + 4 + 4 + 4 + 8 + 8 + 8 + 4;

    pub fn new(dimension: u32) -> Self {
        Header {
            magic: *MAGIC,
            version: VERSION,
            dimension,
            node_count: 0,
            centroid_offset: Self::SIZE as u64,
            index_offset: 0,
            data_offset: 0,
            free_list_head: u32::MAX, // -1 indicates no free nodes
        }
    }

    pub fn serialize(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_le_bytes())?;
        writer.write_all(&self.dimension.to_le_bytes())?;
        writer.write_all(&self.node_count.to_le_bytes())?;
        writer.write_all(&self.centroid_offset.to_le_bytes())?;
        writer.write_all(&self.index_offset.to_le_bytes())?;
        writer.write_all(&self.data_offset.to_le_bytes())?;
        writer.write_all(&self.free_list_head.to_le_bytes())?;
        Ok(())
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(MarsError::InvalidFormat("File too small".into()));
        }

        let magic = [data[0], data[1], data[2], data[3]];
        if &magic != MAGIC {
            return Err(MarsError::InvalidFormat(format!(
                "Invalid magic bytes: expected {:?}, got {:?}",
                MAGIC, magic
            )));
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VERSION {
            return Err(MarsError::InvalidFormat(format!(
                "Unsupported version: {}",
                version
            )));
        }

        Ok(Header {
            magic,
            version,
            dimension: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            node_count: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            centroid_offset: u64::from_le_bytes([
                data[16], data[17], data[18], data[19],
                data[20], data[21], data[22], data[23],
            ]),
            index_offset: u64::from_le_bytes([
                data[24], data[25], data[26], data[27],
                data[28], data[29], data[30], data[31],
            ]),
            data_offset: u64::from_le_bytes([
                data[32], data[33], data[34], data[35],
                data[36], data[37], data[38], data[39],
            ]),
            free_list_head: u32::from_le_bytes([
                data[40], data[41], data[42], data[43],
            ]),
        })
    }
}

/// Index entry for a node
#[derive(Clone, Debug)]
pub struct IndexEntry {
    pub offset: u64,
    pub deleted: bool,
}

impl IndexEntry {
    pub const SIZE: usize = 8 + 1; // offset + deleted flag

    pub fn serialize(&self, writer: &mut impl Write) -> Result<()> {
        writer.write_all(&self.offset.to_le_bytes())?;
        writer.write_all(&[self.deleted as u8])?;
        Ok(())
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(MarsError::InvalidFormat("Index entry too small".into()));
        }

        Ok(IndexEntry {
            offset: u64::from_le_bytes([
                data[0], data[1], data[2], data[3],
                data[4], data[5], data[6], data[7],
            ]),
            deleted: data[8] != 0,
        })
    }
}

/// Storage manager for persistence
pub struct Storage {
    file: File,
    header: Header,
    mmap: Option<MmapMut>,
}

impl Storage {
    /// Create a new database file
    pub fn create<P: AsRef<Path>>(path: P, dimension: u32) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let header = Header::new(dimension);

        let mut storage = Storage {
            file,
            header,
            mmap: None,
        };

        storage.write_header()?;
        storage.initialize_file()?;

        Ok(storage)
    }

    /// Open an existing database file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        let mut header_buf = vec![0u8; Header::SIZE];
        file.read_exact(&mut header_buf)?;

        let header = Header::deserialize(&header_buf)?;

        // Create mmap for reading
        let mmap = unsafe { MmapMut::map_mut(&file).ok() };

        Ok(Storage {
            file,
            header,
            mmap,
        })
    }

    /// Initialize file with centroid space
    fn initialize_file(&mut self) -> Result<()> {
        // Centroid: dimension f32 values initialized to 0
        let centroid_size = self.header.dimension as usize * std::mem::size_of::<f32>();
        let centroid_data = vec![0u8; centroid_size];

        self.file.write_all(&centroid_data)?;

        // Update offsets
        self.header.centroid_offset = Header::SIZE as u64;
        self.header.index_offset = Header::SIZE as u64 + centroid_size as u64;
        self.header.data_offset = self.header.index_offset;

        self.write_header()?;

        Ok(())
    }

    /// Write header to file
    fn write_header(&mut self) -> Result<()> {
        self.file.seek(std::io::SeekFrom::Start(0))?;
        self.header.serialize(&mut self.file)?;
        self.file.flush()?;
        Ok(())
    }

    /// Get the dimension
    pub fn dimension(&self) -> u32 {
        self.header.dimension
    }

    /// Get node count
    pub fn node_count(&self) -> u32 {
        self.header.node_count
    }

    /// Read centroid vector
    pub fn read_centroid(&mut self) -> Result<Vec<f32>> {
        let dimension = self.header.dimension as usize;
        let mut buffer = vec![0u8; dimension * std::mem::size_of::<f32>()];

        self.file.seek(std::io::SeekFrom::Start(self.header.centroid_offset))?;
        self.file.read_exact(&mut buffer)?;

        let centroid = buffer.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(centroid)
    }

    /// Write centroid vector
    pub fn write_centroid(&mut self, centroid: &[f32]) -> Result<()> {
        self.file.seek(std::io::SeekFrom::Start(self.header.centroid_offset))?;

        for &value in centroid {
            self.file.write_all(&value.to_le_bytes())?;
        }

        self.file.flush()?;
        Ok(())
    }

    /// Sync data to disk
    pub fn sync(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_header_serialization() {
        let header = Header::new(128);

        let mut buffer = Vec::new();
        header.serialize(&mut buffer).unwrap();

        assert_eq!(buffer.len(), Header::SIZE);

        let deserialized = Header::deserialize(&buffer).unwrap();
        assert_eq!(deserialized.magic, *MAGIC);
        assert_eq!(deserialized.version, VERSION);
        assert_eq!(deserialized.dimension, 128);
    }

    #[test]
    fn test_storage_create() {
        let temp = NamedTempFile::new().unwrap();
        let storage = Storage::create(temp.path(), 64).unwrap();

        assert_eq!(storage.dimension(), 64);
        assert_eq!(storage.node_count(), 0);
    }

    #[test]
    fn test_storage_centroid() {
        let temp = NamedTempFile::new().unwrap();
        let mut storage = Storage::create(temp.path(), 3).unwrap();

        let centroid = vec![1.0, 2.0, 3.0];
        storage.write_centroid(&centroid).unwrap();

        let read = storage.read_centroid().unwrap();
        assert_eq!(read, centroid);
    }
}
