// Search-side Quoridor position for the AlphaZero engine: the same rules as
// ../game.js, but on flat arrays so MCTS can clone and mutate positions
// cheaply, plus the net's encoding and policy indices. Mirrors
// training/quoridor_core/src/board.rs exactly (checked by
// scripts/check_js_encoding.mjs).
//
// Moves are addressed by policy index (0..140), relative to the side to move.
// Its frame flips rows for player 1 (r -> 8-r, wall anchors r -> 7-r) so the
// mover always heads "up"; columns never flip.
//   0..12    pawn moves by offset in that frame (PAWN_OFFSETS)
//   12..76   h-wall at frame anchor (R, c): 12 + R*8 + c
//   76..140  v-wall at frame anchor (R, c): 76 + R*8 + c

const SIZE = 9
const CELLS = SIZE * SIZE
const WALLS_PER_PLAYER = 10
const H_BASE = 12
const V_BASE = 76
const GOAL_ROW = [SIZE - 1, 0]
const DIRS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
]

export const INPUT_CHANNELS = 8
export const POLICY_SIZE = 140

// N S E W, NN SS EE WW, NE NW SE SW — (dr, dc) in the mover's frame.
const PAWN_OFFSETS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [2, 0],
  [-2, 0],
  [0, 2],
  [0, -2],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
]

const onBoard = (r, c) => r >= 0 && r < SIZE && c >= 0 && c < SIZE

export class Position {
  constructor() {
    this.pawns = [
      [0, 4],
      [SIZE - 1, 4],
    ]
    this.h = new Uint8Array(64) // wall anchors r*8+c
    this.v = new Uint8Array(64)
    this.wallsLeft = [WALLS_PER_PLAYER, WALLS_PER_PLAYER]
    this.turn = 0
    this.winner = null
  }

  // From a game.js state.
  static fromGame(state) {
    const p = new Position()
    p.pawns = state.pawns.map(({ r, c }) => [r, c])
    for (const key of state.walls.keys()) {
      const [r, c] = key.slice(1).split(',').map(Number)
      ;(key[0] === 'h' ? p.h : p.v)[r * 8 + c] = 1
    }
    p.wallsLeft = [...state.wallsLeft]
    p.turn = state.turn
    p.winner = state.winner
    return p
  }

  clone() {
    const p = new Position()
    p.pawns = [[...this.pawns[0]], [...this.pawns[1]]]
    p.h = this.h.slice()
    p.v = this.v.slice()
    p.wallsLeft = [...this.wallsLeft]
    p.turn = this.turn
    p.winner = this.winner
    return p
  }

  hasH(r, c) {
    return r >= 0 && r < 8 && c >= 0 && c < 8 && this.h[r * 8 + c] === 1
  }

  hasV(r, c) {
    return r >= 0 && r < 8 && c >= 0 && c < 8 && this.v[r * 8 + c] === 1
  }

  // True if a one-square step from (r, c) leaves the board or crosses a wall.
  blocked(r, c, dr, dc) {
    if (!onBoard(r + dr, c + dc)) return true
    if (dr !== 0) {
      const row = dr > 0 ? r : r - 1
      return this.hasH(row, c) || this.hasH(row, c - 1)
    }
    const col = dc > 0 ? c : c - 1
    return this.hasV(r, col) || this.hasV(r - 1, col)
  }

  legalPawnTargets() {
    const [mr, mc] = this.pawns[this.turn]
    const [or, oc] = this.pawns[1 - this.turn]
    const out = []
    for (const [dr, dc] of DIRS) {
      if (this.blocked(mr, mc, dr, dc)) continue
      const r = mr + dr
      const c = mc + dc
      if (r !== or || c !== oc) {
        out.push([r, c])
        continue
      }
      if (!this.blocked(r, c, dr, dc)) {
        out.push([r + dr, c + dc])
        continue
      }
      for (const [sr, sc] of dr === 0 ? [[1, 0], [-1, 0]] : [[0, 1], [0, -1]]) {
        if (!this.blocked(r, c, sr, sc)) out.push([r + sr, c + sc])
      }
    }
    return out
  }

  // BFS distance from every cell to `goalRow` (-1 = unreachable).
  distances(goalRow) {
    const dist = new Int16Array(CELLS).fill(-1)
    const queue = new Int16Array(CELLS)
    let head = 0
    let tail = 0
    for (let c = 0; c < SIZE; c++) {
      dist[goalRow * SIZE + c] = 0
      queue[tail++] = goalRow * SIZE + c
    }
    while (head < tail) {
      const id = queue[head++]
      const r = (id / SIZE) | 0
      const c = id % SIZE
      for (const [dr, dc] of DIRS) {
        if (this.blocked(r, c, dr, dc)) continue
        const n = (r + dr) * SIZE + c + dc
        if (dist[n] !== -1) continue
        dist[n] = dist[id] + 1
        queue[tail++] = n
      }
    }
    return dist
  }

  hasPath(player) {
    const [r0, c0] = this.pawns[player]
    const goal = GOAL_ROW[player]
    const seen = new Uint8Array(CELLS)
    const queue = new Int16Array(CELLS)
    let head = 0
    let tail = 0
    seen[r0 * SIZE + c0] = 1
    queue[tail++] = r0 * SIZE + c0
    while (head < tail) {
      const id = queue[head++]
      const r = (id / SIZE) | 0
      const c = id % SIZE
      if (r === goal) return true
      for (const [dr, dc] of DIRS) {
        if (this.blocked(r, c, dr, dc)) continue
        const n = (r + dr) * SIZE + c + dc
        if (seen[n]) continue
        seen[n] = 1
        queue[tail++] = n
      }
    }
    return false
  }

  isLegalWall(horiz, r, c) {
    const conflict = horiz
      ? this.hasH(r, c) || this.hasH(r, c - 1) || this.hasH(r, c + 1) || this.hasV(r, c)
      : this.hasV(r, c) || this.hasV(r - 1, c) || this.hasV(r + 1, c) || this.hasH(r, c)
    if (conflict) return false
    const arr = horiz ? this.h : this.v
    arr[r * 8 + c] = 1
    const ok = this.hasPath(0) && this.hasPath(1)
    arr[r * 8 + c] = 0
    return ok
  }

  pawnIndex(r, c) {
    const [mr, mc] = this.pawns[this.turn]
    const dr = this.turn === 0 ? r - mr : mr - r
    const dc = c - mc
    return PAWN_OFFSETS.findIndex(([a, b]) => a === dr && b === dc)
  }

  wallIndex(horiz, r, c) {
    const fr = this.turn === 0 ? r : 7 - r
    return (horiz ? H_BASE : V_BASE) + fr * 8 + c
  }

  legalMoves() {
    if (this.winner !== null) return []
    const out = this.legalPawnTargets().map(([r, c]) => this.pawnIndex(r, c))
    if (this.wallsLeft[this.turn] > 0) {
      for (const horiz of [true, false]) {
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            if (this.isLegalWall(horiz, r, c)) out.push(this.wallIndex(horiz, r, c))
          }
        }
      }
    }
    return out
  }

  // The game.js move object for a policy index.
  decode(idx) {
    if (idx < H_BASE) {
      const [dr, dc] = PAWN_OFFSETS[idx]
      const [mr, mc] = this.pawns[this.turn]
      return { type: 'pawn', r: mr + (this.turn === 0 ? dr : -dr), c: mc + dc }
    }
    const horiz = idx < V_BASE
    const k = idx - (horiz ? H_BASE : V_BASE)
    const fr = (k / 8) | 0
    return { type: 'wall', r: this.turn === 0 ? fr : 7 - fr, c: k % 8, o: horiz ? 'h' : 'v' }
  }

  // Apply a legal move by policy index (legality is the caller's job).
  push(idx) {
    const me = this.turn
    const m = this.decode(idx)
    if (m.type === 'pawn') {
      this.pawns[me] = [m.r, m.c]
      if (m.r === GOAL_ROW[me]) this.winner = me
    } else {
      ;(m.o === 'h' ? this.h : this.v)[m.r * 8 + m.c] = 1
      this.wallsLeft[me] -= 1
    }
    this.turn = 1 - me
  }

  // −1 for the side to move once the previous mover has won.
  terminalValue() {
    return this.winner !== null ? -1 : null
  }

  // [8, 9, 9] in the side to move's frame — see board.rs `encode_into`.
  encode() {
    const out = new Float32Array(INPUT_CHANNELS * CELLS)
    const me = this.turn
    const flip = me === 1
    const cell = (r, c) => (flip ? SIZE - 1 - r : r) * SIZE + c
    const anchor = (r, c) => (flip ? 7 - r : r) * SIZE + c
    out[cell(...this.pawns[me])] = 1
    out[CELLS + cell(...this.pawns[1 - me])] = 1
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        if (this.h[r * 8 + c]) out[2 * CELLS + anchor(r, c)] = 1
        if (this.v[r * 8 + c]) out[3 * CELLS + anchor(r, c)] = 1
      }
    }
    out.fill(this.wallsLeft[me] / WALLS_PER_PLAYER, 4 * CELLS, 5 * CELLS)
    out.fill(this.wallsLeft[1 - me] / WALLS_PER_PLAYER, 5 * CELLS, 6 * CELLS)
    for (const [plane, player] of [
      [6, me],
      [7, 1 - me],
    ]) {
      const dist = this.distances(GOAL_ROW[player])
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const d = dist[r * SIZE + c]
          out[plane * CELLS + cell(r, c)] = d === -1 ? 2 : Math.min(d, 40) / 20
        }
      }
    }
    return out
  }
}
