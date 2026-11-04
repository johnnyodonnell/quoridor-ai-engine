//! Quoridor rules (2 players, 9×9, 10 walls each) — a port of
//! src/engine/game.js — plus the AlphaZero encoding shared with training/net.py
//! and src/engine/alphazero/encode.js.
//!
//! Coordinates match game.js: r is the row (0 = player 0's home edge), c the
//! column, both 0..8. Player 0 heads for row 8, player 1 for row 0. A wall's
//! anchor (r, c) is in 0..7 with the same geometry as game.js:
//!   h lies between rows r and r+1, covering columns c and c+1.
//!   v lies between columns c and c+1, covering rows r and r+1.
//! Sets of cells are u128 bitboards indexed `r * 9 + c`; walls are u64 masks
//! over their anchors, indexed `r * 8 + c`.
//!
//! Moves are addressed by their policy index (0..140), which is relative to the
//! side to move. Its frame flips rows for player 1 (r -> 8-r, wall anchors
//! r -> 7-r) so the mover always heads "up"; columns never flip.
//!   0..12    pawn moves by offset in that frame (PAWN_OFFSETS)
//!   12..76   h-wall at frame anchor (R, c): 12 + R*8 + c
//!   76..140  v-wall at frame anchor (R, c): 76 + R*8 + c

pub const SIZE: usize = 9;
pub const N_CELLS: usize = SIZE * SIZE;
pub const WALLS_PER_PLAYER: u8 = 10;
pub const INPUT_CHANNELS: usize = 8;
pub const ENC_LEN: usize = INPUT_CHANNELS * N_CELLS;
pub const POLICY_SIZE: usize = 140;
pub const H_BASE: usize = 12;
pub const V_BASE: usize = 76;
/// Games are cut off as a draw after this many plies.
pub const MAX_PLIES: usize = 200;

/// Pawn move offsets (dr, dc) in the mover's frame (+dr = toward its goal),
/// indexed by policy index: N S E W, NN SS EE WW, NE NW SE SW. A diagonal that
/// two routes could produce is the same square, so the mapping is unique.
pub const PAWN_OFFSETS: [(i32, i32); 12] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];
/// The pawn index each pawn index maps to under the left↔right mirror.
const PAWN_MIRROR: [usize; 12] = [0, 1, 3, 2, 4, 5, 7, 6, 9, 8, 11, 10];

const GOAL_ROW: [usize; 2] = [SIZE - 1, 0];
const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

type Cells = u128;
const ALL: Cells = (1u128 << N_CELLS) - 1;

const fn bit(r: usize, c: usize) -> Cells {
    1u128 << (r * SIZE + c)
}
const fn row_mask(r: usize) -> Cells {
    ((1u128 << SIZE) - 1) << (r * SIZE)
}
const fn col_mask(c: usize) -> Cells {
    let mut m = 0;
    let mut r = 0;
    while r < SIZE {
        m |= bit(r, c);
        r += 1;
    }
    m
}

/// Per direction, the cells a pawn can step out of without leaving the board or
/// crossing a wall. `up` is +r.
#[derive(Clone, Copy)]
struct Open {
    up: Cells,
    down: Cells,
    right: Cells,
    left: Cells,
}

impl Open {
    fn new(h: u64, v: u64) -> Self {
        let mut o = Open {
            up: ALL & !row_mask(SIZE - 1),
            down: ALL & !row_mask(0),
            right: ALL & !col_mask(SIZE - 1),
            left: ALL & !col_mask(0),
        };
        for i in bits(h) {
            o.add_wall(true, i / 8, i % 8);
        }
        for i in bits(v) {
            o.add_wall(false, i / 8, i % 8);
        }
        o
    }

    fn add_wall(&mut self, horiz: bool, r: usize, c: usize) {
        if horiz {
            self.up &= !(bit(r, c) | bit(r, c + 1));
            self.down &= !(bit(r + 1, c) | bit(r + 1, c + 1));
        } else {
            self.right &= !(bit(r, c) | bit(r + 1, c));
            self.left &= !(bit(r, c + 1) | bit(r + 1, c + 1));
        }
    }

    fn can(&self, r: i32, c: i32, dr: i32, dc: i32) -> bool {
        let mask = match (dr, dc) {
            (1, 0) => self.up,
            (-1, 0) => self.down,
            (0, 1) => self.right,
            _ => self.left,
        };
        mask & bit(r as usize, c as usize) != 0
    }

    /// Every cell one step from a cell in `s`.
    fn step(&self, s: Cells) -> Cells {
        ((s & self.up) << SIZE)
            | ((s & self.down) >> SIZE)
            | ((s & self.right) << 1)
            | ((s & self.left) >> 1)
    }

    /// Flood fill from `from` until it touches `goal` or stops growing.
    fn reaches(&self, from: Cells, goal: Cells) -> bool {
        let mut reach = from;
        loop {
            if reach & goal != 0 {
                return true;
            }
            let next = reach | self.step(reach);
            if next == reach {
                return false;
            }
            reach = next;
        }
    }

    /// Steps from every cell to the nearest cell of `goal` (255 = unreachable).
    fn distances(&self, goal: Cells) -> [u8; N_CELLS] {
        let mut dist = [u8::MAX; N_CELLS];
        let mut seen = goal;
        let mut frontier = goal;
        let mut d = 0u8;
        while frontier != 0 {
            for i in bits128(frontier) {
                dist[i] = d;
            }
            frontier = self.step(frontier) & !seen;
            seen |= frontier;
            d += 1;
        }
        dist
    }
}

fn bits(mut m: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        Some(i)
    })
}

fn bits128(mut m: u128) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as usize;
        m &= m - 1;
        Some(i)
    })
}

/// A move in absolute board coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Pawn { r: u8, c: u8 },
    Wall { horiz: bool, r: u8, c: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pawns: [(u8, u8); 2],
    h: u64,
    v: u64,
    walls_left: [u8; 2],
    turn: u8,
    winner: Option<u8>,
    ply: u16,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Board {
            pawns: [(0, 4), (SIZE as u8 - 1, 4)],
            h: 0,
            v: 0,
            walls_left: [WALLS_PER_PLAYER; 2],
            turn: 0,
            winner: None,
            ply: 0,
        }
    }

    pub fn turn(&self) -> usize {
        self.turn as usize
    }
    pub fn winner(&self) -> Option<usize> {
        self.winner.map(|w| w as usize)
    }
    pub fn ply(&self) -> usize {
        self.ply as usize
    }
    pub fn pawn(&self, player: usize) -> (usize, usize) {
        let (r, c) = self.pawns[player];
        (r as usize, c as usize)
    }
    pub fn walls_left(&self, player: usize) -> usize {
        self.walls_left[player] as usize
    }
    pub fn has_wall(&self, horiz: bool, r: usize, c: usize) -> bool {
        let m = if horiz { self.h } else { self.v };
        m & (1 << (r * 8 + c)) != 0
    }

    fn open(&self) -> Open {
        Open::new(self.h, self.v)
    }

    /// Squares the side to move may move its pawn to (absolute coordinates).
    pub fn legal_pawn_targets(&self) -> Vec<(usize, usize)> {
        let o = self.open();
        let (mr, mc) = self.pawn(self.turn());
        let (mr, mc) = (mr as i32, mc as i32);
        let (or, oc) = self.pawn(1 - self.turn());
        let opp = (or as i32, oc as i32);
        let mut out = Vec::with_capacity(5);
        for (dr, dc) in DIRS {
            if !o.can(mr, mc, dr, dc) {
                continue;
            }
            let (r, c) = (mr + dr, mc + dc);
            if (r, c) != opp {
                out.push((r as usize, c as usize));
                continue;
            }
            // Opponent is adjacent: jump straight over it if possible, otherwise
            // step diagonally to either side of it.
            if o.can(r, c, dr, dc) {
                out.push(((r + dr) as usize, (c + dc) as usize));
                continue;
            }
            let sides = if dr == 0 { [(1, 0), (-1, 0)] } else { [(0, 1), (0, -1)] };
            for (sr, sc) in sides {
                if o.can(r, c, sr, sc) {
                    out.push(((r + sr) as usize, (c + sc) as usize));
                }
            }
        }
        out
    }

    fn wall_conflicts(&self, horiz: bool, r: usize, c: usize) -> bool {
        let at = |m: u64, r: i32, c: i32| {
            (0..8).contains(&r) && (0..8).contains(&c) && m & (1 << (r * 8 + c)) != 0
        };
        let (r, c) = (r as i32, c as i32);
        if horiz {
            at(self.h, r, c) || at(self.h, r, c - 1) || at(self.h, r, c + 1) || at(self.v, r, c)
        } else {
            at(self.v, r, c) || at(self.v, r - 1, c) || at(self.v, r + 1, c) || at(self.h, r, c)
        }
    }

    fn wall_ok(&self, base: &Open, horiz: bool, r: usize, c: usize) -> bool {
        if self.wall_conflicts(horiz, r, c) {
            return false;
        }
        let mut o = *base;
        o.add_wall(horiz, r, c);
        (0..2).all(|p| {
            let (pr, pc) = self.pawn(p);
            o.reaches(bit(pr, pc), row_mask(GOAL_ROW[p]))
        })
    }

    pub fn is_legal_wall(&self, horiz: bool, r: usize, c: usize) -> bool {
        if self.winner.is_some() || self.walls_left[self.turn()] == 0 || r >= 8 || c >= 8 {
            return false;
        }
        self.wall_ok(&self.open(), horiz, r, c)
    }

    /// Legal walls for the side to move as (horiz, r, c), h before v, row-major
    /// (game.js `legalWalls` order).
    pub fn legal_walls(&self) -> Vec<(bool, usize, usize)> {
        let mut out = Vec::new();
        if self.winner.is_some() || self.walls_left[self.turn()] == 0 {
            return out;
        }
        let base = self.open();
        for horiz in [true, false] {
            for r in 0..8 {
                for c in 0..8 {
                    if self.wall_ok(&base, horiz, r, c) {
                        out.push((horiz, r, c));
                    }
                }
            }
        }
        out
    }

    /// Policy indices of every legal move (empty once the game is won).
    pub fn legal_moves(&self) -> Vec<u16> {
        if self.winner.is_some() {
            return Vec::new();
        }
        let mut out: Vec<u16> = self
            .legal_pawn_targets()
            .into_iter()
            .map(|(r, c)| self.pawn_index(r, c))
            .collect();
        out.extend(self.legal_walls().into_iter().map(|(h, r, c)| self.wall_index(h, r, c)));
        out
    }

    /// Policy index of moving the side to move's pawn to (r, c).
    pub fn pawn_index(&self, r: usize, c: usize) -> u16 {
        let (mr, mc) = self.pawn(self.turn());
        let dr = r as i32 - mr as i32;
        let dr = if self.turn == 0 { dr } else { -dr };
        let off = (dr, c as i32 - mc as i32);
        PAWN_OFFSETS
            .iter()
            .position(|&o| o == off)
            .unwrap_or_else(|| panic!("no pawn index for offset {off:?}")) as u16
    }

    /// Policy index of the side to move placing a wall at absolute anchor (r, c).
    pub fn wall_index(&self, horiz: bool, r: usize, c: usize) -> u16 {
        let fr = if self.turn == 0 { r } else { 7 - r };
        ((if horiz { H_BASE } else { V_BASE }) + fr * 8 + c) as u16
    }

    pub fn move_index(&self, m: Move) -> u16 {
        match m {
            Move::Pawn { r, c } => self.pawn_index(r as usize, c as usize),
            Move::Wall { horiz, r, c } => self.wall_index(horiz, r as usize, c as usize),
        }
    }

    /// The absolute move a policy index means for the side to move.
    pub fn decode(&self, idx: u16) -> Move {
        let idx = idx as usize;
        if idx < H_BASE {
            let (dr, dc) = PAWN_OFFSETS[idx];
            let dr = if self.turn == 0 { dr } else { -dr };
            let (mr, mc) = self.pawn(self.turn());
            return Move::Pawn {
                r: (mr as i32 + dr) as u8,
                c: (mc as i32 + dc) as u8,
            };
        }
        let horiz = idx < V_BASE;
        let k = idx - if horiz { H_BASE } else { V_BASE };
        let (fr, c) = (k / 8, k % 8);
        let r = if self.turn == 0 { fr } else { 7 - fr };
        Move::Wall {
            horiz,
            r: r as u8,
            c: c as u8,
        }
    }

    /// Apply a legal move given by policy index. Legality is the caller's job
    /// (MCTS only pushes indices taken from `legal_moves`).
    pub fn push(&mut self, idx: u16) {
        let me = self.turn();
        match self.decode(idx) {
            Move::Pawn { r, c } => {
                self.pawns[me] = (r, c);
                if r as usize == GOAL_ROW[me] {
                    self.winner = Some(me as u8);
                }
            }
            Move::Wall { horiz, r, c } => {
                let b = 1u64 << (r as usize * 8 + c as usize);
                if horiz {
                    self.h |= b;
                } else {
                    self.v |= b;
                }
                self.walls_left[me] -= 1;
            }
        }
        self.turn ^= 1;
        self.ply += 1;
    }

    /// Value for the side to move if the game is over: −1 when the previous
    /// mover just won, 0 at the ply cap.
    pub fn terminal_value(&self) -> Option<f32> {
        if self.winner.is_some() {
            Some(-1.0)
        } else if self.ply() >= MAX_PLIES {
            Some(0.0)
        } else {
            None
        }
    }

    /// Result from player 0's POV (+1 / −1 / 0 for the ply-cap draw), or None
    /// while the game is still going.
    pub fn outcome(&self) -> Option<i8> {
        match self.winner {
            Some(0) => Some(1),
            Some(_) => Some(-1),
            None if self.ply() >= MAX_PLIES => Some(0),
            None => None,
        }
    }

    /// Encode into `out` (len ENC_LEN, [8, 9, 9]) in the side to move's frame:
    ///   0 my pawn · 1 opp pawn · 2 h-wall anchors · 3 v-wall anchors ·
    ///   4 my walls left / 10 · 5 opp walls left / 10 ·
    ///   6 per-cell distance to my goal · 7 per-cell distance to opp goal
    /// Distances are min(d, 40) / 20, and 2.0 for cells walled off entirely.
    pub fn encode_into(&self, out: &mut [f32]) {
        assert_eq!(out.len(), ENC_LEN);
        out.fill(0.0);
        let me = self.turn();
        let flip = me == 1;
        let cell = |r: usize, c: usize| (if flip { SIZE - 1 - r } else { r }) * SIZE + c;
        let anchor = |r: usize, c: usize| (if flip { 7 - r } else { r }) * SIZE + c;
        let plane = |p: usize| p * N_CELLS;

        let (r, c) = self.pawn(me);
        out[plane(0) + cell(r, c)] = 1.0;
        let (r, c) = self.pawn(1 - me);
        out[plane(1) + cell(r, c)] = 1.0;
        for i in bits(self.h) {
            out[plane(2) + anchor(i / 8, i % 8)] = 1.0;
        }
        for i in bits(self.v) {
            out[plane(3) + anchor(i / 8, i % 8)] = 1.0;
        }
        out[plane(4)..plane(5)].fill(self.walls_left[me] as f32 / WALLS_PER_PLAYER as f32);
        out[plane(5)..plane(6)].fill(self.walls_left[1 - me] as f32 / WALLS_PER_PLAYER as f32);

        let o = self.open();
        for (p, player) in [(6, me), (7, 1 - me)] {
            let dist = o.distances(row_mask(GOAL_ROW[player]));
            for r in 0..SIZE {
                for c in 0..SIZE {
                    let d = dist[r * SIZE + c];
                    out[plane(p) + cell(r, c)] = if d == u8::MAX {
                        2.0
                    } else {
                        d.min(40) as f32 / 20.0
                    };
                }
            }
        }
    }

    pub fn encode(&self) -> Vec<f32> {
        let mut out = vec![0.0; ENC_LEN];
        self.encode_into(&mut out);
        out
    }

    /// This position reflected left↔right (column c -> 8-c, wall anchors c -> 7-c).
    pub fn mirrored(&self) -> Board {
        let mirror = |m: u64| bits(m).fold(0u64, |acc, i| acc | 1 << ((i / 8) * 8 + 7 - i % 8));
        Board {
            pawns: self.pawns.map(|(r, c)| (r, SIZE as u8 - 1 - c)),
            h: mirror(self.h),
            v: mirror(self.v),
            ..self.clone()
        }
    }
}

/// Gather indices that reflect an encoding left↔right: `mirrored[i] = enc[perm[i]]`.
/// Cell planes map column c -> 8-c; wall-anchor planes map c -> 7-c (their
/// always-empty column 8 maps to itself).
pub fn mirror_state_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(ENC_LEN);
    for p in 0..INPUT_CHANNELS {
        let wall_plane = p == 2 || p == 3;
        for r in 0..SIZE {
            for c in 0..SIZE {
                let mc = if !wall_plane {
                    SIZE - 1 - c
                } else if c < 8 {
                    7 - c
                } else {
                    c
                };
                perm.push(p * N_CELLS + r * SIZE + mc);
            }
        }
    }
    perm
}

/// Gather indices that reflect a policy vector left↔right: `mirrored[i] = pi[perm[i]]`.
pub fn mirror_policy_perm() -> Vec<usize> {
    let mut perm: Vec<usize> = PAWN_MIRROR.to_vec();
    for base in [H_BASE, V_BASE] {
        for r in 0..8 {
            for c in 0..8 {
                perm.push(base + r * 8 + 7 - c);
            }
        }
    }
    perm
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_board(rng: &mut StdRng, plies: usize) -> Board {
        let mut b = Board::new();
        for _ in 0..plies {
            let moves = b.legal_moves();
            if moves.is_empty() {
                break;
            }
            b.push(moves[rng.gen_range(0..moves.len())]);
        }
        b
    }

    #[test]
    fn opening_moves() {
        let b = Board::new();
        assert_eq!(b.legal_pawn_targets().len(), 3);
        assert_eq!(b.legal_walls().len(), 128);
        let moves = b.legal_moves();
        assert_eq!(moves.len(), 131);
        assert!(moves.iter().all(|&m| (m as usize) < POLICY_SIZE));
    }

    #[test]
    fn index_roundtrip() {
        let mut rng = StdRng::seed_from_u64(1);
        for plies in 0..60 {
            let b = random_board(&mut rng, plies);
            for idx in b.legal_moves() {
                assert_eq!(b.move_index(b.decode(idx)), idx);
            }
        }
    }

    #[test]
    fn both_players_head_up_in_their_frame() {
        // N (index 0) moves player 0 to row 1 and player 1 to row 7.
        let mut b = Board::new();
        assert_eq!(b.decode(0), Move::Pawn { r: 1, c: 4 });
        b.push(0);
        assert_eq!(b.decode(0), Move::Pawn { r: 7, c: 4 });
    }

    #[test]
    fn perms_are_involutions() {
        for perm in [mirror_state_perm(), mirror_policy_perm()] {
            for (i, &j) in perm.iter().enumerate() {
                assert_eq!(perm[j], i);
            }
        }
    }

    #[test]
    fn mirror_perms_match_mirrored_board() {
        let sp = mirror_state_perm();
        let pp = mirror_policy_perm();
        let mut rng = StdRng::seed_from_u64(2);
        for plies in 0..80 {
            let b = random_board(&mut rng, plies);
            let m = b.mirrored();
            let (e, me) = (b.encode(), m.encode());
            for i in 0..ENC_LEN {
                assert_eq!(me[i], e[sp[i]]);
            }
            let mut want: Vec<u16> = b.legal_moves().iter().map(|&i| pp[i as usize] as u16).collect();
            let mut got = m.legal_moves();
            want.sort();
            got.sort();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn jump_and_diagonals() {
        let mut b = Board::new();
        b.pawns = [(4, 4), (5, 4)];
        let mut t = b.legal_pawn_targets();
        t.sort();
        assert_eq!(t, vec![(3, 4), (4, 3), (4, 5), (6, 4)]);
        // Wall behind the opponent: the jump turns into two diagonals.
        b.h |= 1 << (5 * 8 + 4);
        let mut t = b.legal_pawn_targets();
        t.sort();
        assert_eq!(t, vec![(3, 4), (4, 3), (4, 5), (5, 3), (5, 5)]);
    }

    #[test]
    fn walls_cannot_seal_a_pawn() {
        let mut b = Board::new();
        // h walls at row 0 over columns 0-1, 2-3, 4-5, 6-7 leave only column 8 open.
        for c in [0, 2, 4, 6] {
            b.h |= 1 << c;
        }
        // A v wall between columns 7 and 8 on rows 0-1 seals player 0 in.
        assert!(!b.is_legal_wall(false, 0, 7));
        assert!(b.is_legal_wall(false, 1, 7));
    }

    #[test]
    fn winning_and_terminal_value() {
        let mut b = Board::new();
        b.pawns[0] = (7, 0);
        b.push(b.pawn_index(8, 0));
        assert_eq!(b.winner(), Some(0));
        assert_eq!(b.terminal_value(), Some(-1.0));
        assert_eq!(b.outcome(), Some(1));
        assert!(b.legal_moves().is_empty());
    }
}
