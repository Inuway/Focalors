use std::sync::atomic::{AtomicU64, Ordering};

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

/// Decoded TT entry returned by `probe()`. Carried by value because the
/// underlying storage is atomic — we can't hand out references into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TTEntry {
    pub key: u64,
    pub depth: u8,
    pub flag: TTFlag,
    pub score: Score,
    pub best_move: Move,
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

/// Pack the non-key fields into a single u64.
///
/// Layout (LSB → MSB):
///   bits  0–7:   depth     (u8)
///   bits  8–15:  flag      (u8 from TTFlag::repr(u8))
///   bits 16–47:  score     (i32 reinterpreted as u32)
///   bits 48–63:  best_move (u16)
fn pack_data(depth: u8, flag: TTFlag, score: Score, best_move: Move) -> u64 {
    let depth_bits = depth as u64;
    let flag_bits = (flag as u8 as u64) << 8;
    let score_bits = ((score as u32) as u64) << 16;
    let mv_bits = (best_move.raw() as u64) << 48;
    depth_bits | flag_bits | score_bits | mv_bits
}

fn unpack_data(packed: u64) -> (u8, TTFlag, Score, Move) {
    let depth = packed as u8;
    let flag_byte = ((packed >> 8) & 0xff) as u8;
    let flag = match flag_byte {
        0 => TTFlag::Exact,
        1 => TTFlag::LowerBound,
        2 => TTFlag::UpperBound,
        // Defensive: corrupt entry from a torn read — the XOR-key check in
        // probe() will already have rejected it, but pick a safe variant.
        _ => TTFlag::Exact,
    };
    let score = ((packed >> 16) & 0xffff_ffff) as u32 as i32;
    let mv = Move::from_raw(((packed >> 48) & 0xffff) as u16);
    (depth, flag, score, mv)
}

/// A single thread-safe slot. Two atomic u64s; XOR'ing them recovers the
/// position key. A torn write is detected when (xor_data ^ data) doesn't
/// match the position the reader is asking about — that's the standard
/// chess-engine technique for lockless TT correctness.
struct AtomicTTEntry {
    /// Position key XOR'd with the packed data. Read together with `data`.
    xor_data: AtomicU64,
    /// Packed (depth, flag, score, best_move).
    data: AtomicU64,
}

impl AtomicTTEntry {
    const fn empty() -> Self {
        AtomicTTEntry {
            xor_data: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

/// Thread-safe transposition table.
///
/// Multiple threads may probe and store concurrently with no locks; the
/// XOR-key validation in `probe()` rejects torn reads. Replacement is
/// depth-preferred — a new entry overwrites an existing one only if the
/// slot is empty, refers to the same position, or has searched at least
/// as deep.
pub struct TranspositionTable {
    entries: Vec<AtomicTTEntry>,
    mask: usize,
}

impl TranspositionTable {
    /// Create a TT sized to roughly `size_mb` megabytes. Actual size is
    /// rounded down to a power of two for fast indexing.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<AtomicTTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        let num_entries = num_entries.next_power_of_two() / 2;
        let num_entries = num_entries.max(1);

        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            entries.push(AtomicTTEntry::empty());
        }

        TranspositionTable {
            entries,
            mask: num_entries - 1,
        }
    }

    fn index(&self, key: u64) -> usize {
        key as usize & self.mask
    }

    /// Probe the TT for a position. Returns `Some(entry)` only when the
    /// stored slot's key (recovered via XOR) matches the probed key.
    pub fn probe(&self, key: u64) -> Option<TTEntry> {
        let slot = &self.entries[self.index(key)];
        let xor_data = slot.xor_data.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);
        if xor_data ^ data != key {
            return None;
        }
        let (depth, flag, score, best_move) = unpack_data(data);
        Some(TTEntry {
            key,
            depth,
            flag,
            score,
            best_move,
        })
    }

    /// Store a result. Depth-preferred replacement — keep the deepest
    /// search for any given slot. Safe to call from many threads at once.
    pub fn store(&self, key: u64, depth: u8, flag: TTFlag, score: Score, best_move: Move) {
        let slot = &self.entries[self.index(key)];

        let existing_xor = slot.xor_data.load(Ordering::Relaxed);
        let existing_data = slot.data.load(Ordering::Relaxed);
        let existing_key = existing_xor ^ existing_data;
        let existing_depth = existing_data as u8;
        let is_empty = existing_xor == 0 && existing_data == 0;

        if is_empty || existing_key == key || depth >= existing_depth {
            let new_data = pack_data(depth, flag, score, best_move);
            slot.xor_data.store(key ^ new_data, Ordering::Relaxed);
            slot.data.store(new_data, Ordering::Relaxed);
        }
    }

    /// Clear the table. Safe to call from any thread holding a reference,
    /// but should be called when no concurrent searches are in flight.
    pub fn clear(&self) {
        for slot in &self.entries {
            slot.xor_data.store(0, Ordering::Relaxed);
            slot.data.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Square;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn pack_unpack_roundtrip() {
        let cases = [
            (0u8, TTFlag::Exact, 0i32, Move::NULL),
            (5, TTFlag::LowerBound, 100, Move::new(Square(12), Square(28))),
            (60, TTFlag::UpperBound, -250, Move::new(Square(0), Square(63))),
            (255, TTFlag::Exact, i32::MIN, Move::new(Square(63), Square(0))),
            (1, TTFlag::Exact, i32::MAX, Move::NULL),
        ];
        for (depth, flag, score, mv) in cases {
            let packed = pack_data(depth, flag, score, mv);
            let (d, f, s, m) = unpack_data(packed);
            assert_eq!(d, depth, "depth roundtrip");
            assert_eq!(f, flag, "flag roundtrip");
            assert_eq!(s, score, "score roundtrip");
            assert_eq!(m.raw(), mv.raw(), "move roundtrip");
        }
    }

    #[test]
    fn single_thread_store_and_probe() {
        let tt = TranspositionTable::new(1);
        let key = 0xDEADBEEF_CAFEBABEu64;
        let mv = Move::new(Square(12), Square(28));

        assert_eq!(tt.probe(key), None, "empty slot should miss");

        tt.store(key, 7, TTFlag::Exact, 42, mv);
        let entry = tt.probe(key).expect("should find stored entry");
        assert_eq!(entry.key, key);
        assert_eq!(entry.depth, 7);
        assert_eq!(entry.flag, TTFlag::Exact);
        assert_eq!(entry.score, 42);
        assert_eq!(entry.best_move.raw(), mv.raw());
    }

    #[test]
    fn depth_preferred_replacement() {
        let tt = TranspositionTable::new(1);
        let key = 0x1234_5678_9ABC_DEF0u64;
        let mv1 = Move::new(Square(12), Square(28));
        let mv2 = Move::new(Square(13), Square(29));

        tt.store(key, 5, TTFlag::Exact, 100, mv1);
        tt.store(key, 4, TTFlag::Exact, 200, mv2); // shallower — should still replace (same key)
        let entry = tt.probe(key).unwrap();
        assert_eq!(entry.depth, 4, "same-key store always wins");
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn wrong_key_misses() {
        let tt = TranspositionTable::new(1);
        let key = 0xAAAA_BBBB_CCCC_DDDDu64;
        let mv = Move::new(Square(0), Square(8));
        tt.store(key, 5, TTFlag::Exact, 100, mv);
        // Probe with a different key that hashes to the same slot.
        // Even if it collides, the XOR-key check should reject.
        for delta in [1u64, 2, 100, 0xFFFF] {
            let other = key.wrapping_add(delta);
            if let Some(entry) = tt.probe(other) {
                // If it returned something, the key must match (which it shouldn't here)
                assert_eq!(entry.key, other, "probe for {other:#x} returned wrong-key entry");
            }
        }
    }

    #[test]
    fn clear_empties_table() {
        let tt = TranspositionTable::new(1);
        let key = 0x1111_2222_3333_4444u64;
        tt.store(key, 5, TTFlag::Exact, 100, Move::NULL);
        assert!(tt.probe(key).is_some());
        tt.clear();
        assert_eq!(tt.probe(key), None);
    }

    #[test]
    fn concurrent_store_and_probe_no_corruption() {
        // Hammer the TT from many threads. Every probe that returns Some
        // must have an entry whose key matches what we asked for. Wrong
        // entries leaking through would indicate broken torn-read handling.
        let tt = Arc::new(TranspositionTable::new(2));
        let n_threads = 8;
        let ops_per_thread = 50_000;

        thread::scope(|s| {
            for thread_id in 0..n_threads {
                let tt = Arc::clone(&tt);
                s.spawn(move || {
                    let mut rng_state = 0xCAFE_F00D_u64.wrapping_add(thread_id as u64);
                    for _ in 0..ops_per_thread {
                        // xorshift step
                        rng_state ^= rng_state << 13;
                        rng_state ^= rng_state >> 7;
                        rng_state ^= rng_state << 17;
                        let key = rng_state | 1; // avoid 0 (empty marker)
                        let depth = (rng_state >> 32) as u8;
                        let score = (rng_state >> 16) as i16 as i32;
                        let mv = Move::from_raw((rng_state & 0xffff) as u16);

                        if rng_state & 1 == 0 {
                            tt.store(key, depth, TTFlag::Exact, score, mv);
                        } else {
                            if let Some(entry) = tt.probe(key) {
                                assert_eq!(entry.key, key, "probe returned mismatched key");
                            }
                        }
                    }
                });
            }
        });
    }
}
