// Checks the browser engine's Position (src/engine/alphazero/position.js)
// against the Rust board: replays games dumped by
// training/dump_encoding_fixture.py and compares legal moves, move decoding
// and the [8, 9, 9] encoding at every ply.
//
//   node scripts/check_js_encoding.mjs [training/fixtures/encoding.json]

import { readFileSync } from 'node:fs'

import { Position } from '../src/engine/alphazero/position.js'

const path = process.argv[2] ?? 'training/fixtures/encoding.json'
const games = JSON.parse(readFileSync(path, 'utf8'))

let plies = 0
for (const [gi, game] of games.entries()) {
  const pos = new Position()
  for (const [pi, ply] of game.plies.entries()) {
    const at = `game ${gi} ply ${pi}`
    const legal = pos.legalMoves().sort((a, b) => a - b)
    if (JSON.stringify(legal) !== JSON.stringify(ply.legal)) {
      throw new Error(`${at}: legal moves differ\n js:   ${legal}\n rust: ${ply.legal}`)
    }
    const enc = pos.encode()
    for (let i = 0; i < enc.length; i++) {
      if (Math.abs(enc[i] - ply.enc[i]) > 1e-5) {
        throw new Error(`${at}: encoding differs at ${i}: js ${enc[i]} rust ${ply.enc[i]}`)
      }
    }
    const m = pos.decode(ply.move)
    const [kind, r, c] = ply.decoded
    const want = kind === 'pawn' ? { type: 'pawn', r, c } : { type: 'wall', r, c, o: kind }
    if (JSON.stringify(m) !== JSON.stringify(want)) {
      throw new Error(`${at}: decode(${ply.move}) js ${JSON.stringify(m)} rust ${JSON.stringify(want)}`)
    }
    pos.push(ply.move)
    plies++
  }
  const winner = pos.winner === null ? null : pos.winner === 0 ? 1 : -1
  if (game.outcome !== 0 && winner !== game.outcome) {
    throw new Error(`game ${gi}: outcome js ${winner} rust ${game.outcome}`)
  }
}
console.log(`ok: ${games.length} games, ${plies} plies match`)
