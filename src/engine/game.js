// Pure Quoridor rules (2 players, 9×9, 10 walls each). No React, no side
// effects: `applyMove` returns a new state rather than mutating its input.
//
// Coordinates: r is the row (0 = the human's home edge, drawn at the bottom),
// c is the column, both 0..8.
//
// A wall is { r, c, o } with o = 'h' | 'v' and anchor r, c in 0..7 — the
// square nearest a1 that the wall touches, as in standard notation:
//   'h' lies between rows r and r+1, covering columns c and c+1.
//   'v' lies between columns c and c+1, covering rows r and r+1.
// Walls may not overlap, may not cross at their midpoint, and may not cut
// either pawn off from its goal row.
//
// A move is { type: 'pawn', r, c } or { type: 'wall', r, c, o }.

export const HUMAN = 0
export const BOT = 1

const SIZE = 9
const WALLS_PER_PLAYER = 10
const GOAL_ROW = [SIZE - 1, 0]
const DIRS = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
]

const wallKey = (o, r, c) => `${o}${r},${c}`

export function createGame() {
  return {
    pawns: [
      { r: 0, c: 4 },
      { r: SIZE - 1, c: 4 },
    ],
    walls: new Set(),
    wallsLeft: [WALLS_PER_PLAYER, WALLS_PER_PLAYER],
    turn: HUMAN,
    winner: null,
    // { player, move, from } — `from` is the pawn's square before a pawn move.
    lastMove: null,
  }
}

const onBoard = (r, c) => r >= 0 && r < SIZE && c >= 0 && c < SIZE

// True if a one-square step from (r, c) in direction (dr, dc) leaves the board
// or passes through a wall.
function blocked(walls, r, c, dr, dc) {
  if (!onBoard(r + dr, c + dc)) return true
  if (dr !== 0) {
    const row = dr > 0 ? r : r - 1
    return walls.has(wallKey('h', row, c)) || walls.has(wallKey('h', row, c - 1))
  }
  const col = dc > 0 ? c : c - 1
  return walls.has(wallKey('v', r, col)) || walls.has(wallKey('v', r - 1, col))
}

// Squares the side to move may move its pawn to.
export function legalPawnMoves(state) {
  const me = state.pawns[state.turn]
  const opp = state.pawns[1 - state.turn]
  const moves = []
  for (const [dr, dc] of DIRS) {
    if (blocked(state.walls, me.r, me.c, dr, dc)) continue
    const r = me.r + dr
    const c = me.c + dc
    if (r !== opp.r || c !== opp.c) {
      moves.push({ r, c })
      continue
    }
    // Opponent is adjacent: jump straight over it if possible, otherwise
    // step diagonally to either side of it.
    if (!blocked(state.walls, r, c, dr, dc)) {
      moves.push({ r: r + dr, c: c + dc })
      continue
    }
    for (const [sr, sc] of dr === 0 ? [[1, 0], [-1, 0]] : [[0, 1], [0, -1]]) {
      if (!blocked(state.walls, r, c, sr, sc)) moves.push({ r: r + sr, c: c + sc })
    }
  }
  return moves
}

// BFS from (r, c) to any square on `goalRow`. Pawns don't block paths.
function hasPath(walls, { r, c }, goalRow) {
  const seen = new Set([r * SIZE + c])
  const queue = [[r, c]]
  while (queue.length) {
    const [qr, qc] = queue.shift()
    if (qr === goalRow) return true
    for (const [dr, dc] of DIRS) {
      if (blocked(walls, qr, qc, dr, dc)) continue
      const id = (qr + dr) * SIZE + qc + dc
      if (seen.has(id)) continue
      seen.add(id)
      queue.push([qr + dr, qc + dc])
    }
  }
  return false
}

export function isLegalWall(state, { r, c, o }) {
  if (state.winner !== null || state.wallsLeft[state.turn] === 0) return false
  if (r < 0 || r >= SIZE - 1 || c < 0 || c >= SIZE - 1) return false
  const { walls } = state
  // Overlap with a parallel wall at the same anchor or one step along its
  // length, or a crossing wall sharing its midpoint.
  const conflicts =
    o === 'h'
      ? [wallKey('h', r, c), wallKey('h', r, c - 1), wallKey('h', r, c + 1), wallKey('v', r, c)]
      : [wallKey('v', r, c), wallKey('v', r - 1, c), wallKey('v', r + 1, c), wallKey('h', r, c)]
  if (conflicts.some((k) => walls.has(k))) return false
  const next = new Set(walls).add(wallKey(o, r, c))
  return state.pawns.every((p, i) => hasPath(next, p, GOAL_ROW[i]))
}

export function legalWalls(state) {
  const result = []
  for (const o of ['h', 'v']) {
    for (let r = 0; r < SIZE - 1; r++) {
      for (let c = 0; c < SIZE - 1; c++) {
        if (isLegalWall(state, { r, c, o })) result.push({ type: 'wall', r, c, o })
      }
    }
  }
  return result
}

// Returns the state after `move`, or null if the move is illegal.
export function applyMove(state, move) {
  if (state.winner !== null) return null
  const player = state.turn
  const next = { ...state, turn: 1 - player, lastMove: { player, move, from: state.pawns[player] } }

  if (move.type === 'pawn') {
    if (!legalPawnMoves(state).some((m) => m.r === move.r && m.c === move.c)) return null
    next.pawns = state.pawns.map((p, i) => (i === player ? { r: move.r, c: move.c } : p))
    if (move.r === GOAL_ROW[player]) next.winner = player
    return next
  }

  if (!isLegalWall(state, move)) return null
  next.walls = new Set(state.walls).add(wallKey(move.o, move.r, move.c))
  next.wallsLeft = state.wallsLeft.map((n, i) => (i === player ? n - 1 : n))
  return next
}

export function getTurn(state) {
  return state.turn
}

export function isGameOver(state) {
  return state.winner !== null
}

// `kind` is 'in-progress' or 'win' (with `winner`). Quoridor has no draws.
export function getStatus(state) {
  if (state.winner !== null) return { kind: 'win', winner: state.winner }
  return { kind: 'in-progress' }
}
