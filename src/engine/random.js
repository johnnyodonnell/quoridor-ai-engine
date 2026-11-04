// The v1 AI engine: a random legal move, chosen in two stages so the bot
// doesn't burn all its walls immediately (there are ~100 legal walls vs ~4
// pawn moves). First a coin flip between "wall" and "pawn", then a uniform
// pick within that kind. Falls back to a pawn move when no wall is legal.
//
// Contract — same shape as chess-ai-engine/src/engine/random.js:
//   bestMove(state) -> a move object (see game.js)
// Caller must only invoke when it's the bot's turn and the game isn't over.

import { legalPawnMoves, legalWalls } from './game.js'

const pick = (xs) => xs[Math.floor(Math.random() * xs.length)]

export function bestMove(state) {
  if (state.wallsLeft[state.turn] > 0 && Math.random() < 0.5) {
    const walls = legalWalls(state)
    if (walls.length) return pick(walls)
  }
  return { type: 'pawn', ...pick(legalPawnMoves(state)) }
}
