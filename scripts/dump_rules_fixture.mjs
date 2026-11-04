// Plays seeded random games through src/engine/game.js and dumps every ply's
// state, legal pawn moves and legal walls as JSON, so the Rust rules engine
// (training/quoridor_core) can be checked against the JS rules.
//
//   node scripts/dump_rules_fixture.mjs [nGames=500] [out=training/fixtures/rules.json]
//
// Pawn moves lean toward the goal so pawns meet often and jumps/diagonals get
// exercised; walls are placed at random about a third of the time.

import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import { applyMove, createGame, legalPawnMoves, legalWalls, wallList } from '../src/engine/game.js'

const nGames = Number(process.argv[2] ?? 500)
const out = process.argv[3] ?? 'training/fixtures/rules.json'

// mulberry32: small seeded PRNG so the fixture is reproducible.
let seed = 12345
function rand() {
  seed = (seed + 0x6d2b79f5) | 0
  let t = seed
  t = Math.imul(t ^ (t >>> 15), t | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const pick = (xs) => xs[Math.floor(rand() * xs.length)]

const sq = ({ r, c }) => `${r},${c}`
const wall = ({ r, c, o }) => `${o}${r},${c}`

function chooseMove(state, pawn, walls) {
  if (walls.length && rand() < 0.35) return pick(walls)
  const forward = state.turn === 0 ? 1 : -1
  const best = Math.max(...pawn.map((m) => m.r * forward))
  const toward = pawn.filter((m) => m.r * forward === best)
  return { type: 'pawn', ...(rand() < 0.7 ? pick(toward) : pick(pawn)) }
}

const games = []
for (let g = 0; g < nGames; g++) {
  let state = createGame()
  const plies = []
  while (state.winner === null && plies.length < 300) {
    const pawn = legalPawnMoves(state)
    const walls = legalWalls(state)
    const move = chooseMove(state, pawn, walls)
    plies.push({
      pawns: state.pawns.map(sq),
      walls: wallList(state).map(wall).sort(),
      wallsLeft: state.wallsLeft,
      turn: state.turn,
      pawn: pawn.map(sq).sort(),
      legalWalls: walls.map(wall).sort(),
      move: move.type === 'pawn' ? `p${sq(move)}` : wall(move),
    })
    state = applyMove(state, move)
  }
  games.push({ plies, winner: state.winner })
}

mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, JSON.stringify(games))
const total = games.reduce((n, g) => n + g.plies.length, 0)
console.log(`wrote ${out}: ${games.length} games, ${total} plies`)
