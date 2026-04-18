use crate::eval::Score;
use crate::moves::Move;

/// What kind of score is stored in this TT entry?
///
/// During alpha-beta search, we don't always get the exact score:
/// - **Exact**: we searched all moves and know the true value
/// - **LowerBound** (beta cutoff): the score is *at least* this good
/// - **UpperBound** (failed low): the score is *at most* this good
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TTFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

/// A single entry in the transposition table.
#[derive(Clone, Copy)]
pub struct TTEntry {
    pub key: u64,     // Zobrist hash to verify this is the right position
    pub depth: u8,    // Search depth this result comes from
    pub flag: TTFlag, // Type of bound
    pub score: Score,
    pub best_move: Move, // Best move found (useful for move ordering)
}

impl Default for TTEntry {
    fn default() -> Self {
        TTEntry {
            key: 0,
            depth: 0,
            flag: TTFlag::Exact,
            score: 0,
            best_move: Move::NULL,
        }
    }
}

/// The transposition table: a large hash map from position -> search result.
///
/// Uses a simple replacement scheme: always replace (depth-preferred would be
/// better but this is simpler and still very effective).
pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    mask: usize, // size - 1 (for fast modulo since size is power of 2)
}

impl TranspositionTable {
    /// Create a TT with the given size in megabytes.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        // Round down to power of 2 for fast indexing
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1);

        TranspositionTable {
            entries: vec![TTEntry::default(); num_entries],
            mask: num_entries - 1,
        }
    }

    fn index(&self, key: u64) -> usize {
        key as usize & self.mask
    }

    /// Probe the TT for a position.
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let entry = &self.entries[self.index(key)];
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    /// Store a result in the TT. Uses always-replace strategy with
    /// depth preference (only replace if new depth >= stored depth).
    pub fn store(&mut self, key: u64, depth: u8, flag: TTFlag, score: Score, best_move: Move) {
        let idx = self.index(key);
        let existing = &self.entries[idx];

        // Replace if: empty slot, same position, or deeper/equal search
        if existing.key == 0 || existing.key == key || depth >= existing.depth {
            self.entries[idx] = TTEntry {
                key,
                depth,
                flag,
                score,
                best_move,
            };
        }
    }

    /// Clear the table (e.g. for a new game).
    pub fn clear(&mut self) {
        self.entries.fill(TTEntry::default());
    }
}
